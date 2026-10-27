import json
from collections import defaultdict
from pathlib import Path

import awkward as ak
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

from utils.parquet import read_parquet_files


class HGCPredictionWriter(Callback):
    """Writes MaskFormer test-time predictions to a parquet file with the same schema as the
    dataset's input file, plus extra prediction branches.

    For each event, decoder outputs are permuted by the model's Hungarian matcher so that
    prediction slot `i` corresponds to true particle `i` for `i < n_particles` (see
    `MaskFormer.loss`). This lets us attach one prediction per truth particle directly onto the
    particle collection. Slots `i >= n_particles` are extra/unmatched query slots (e.g. fakes);
    we don't keep per-fake detail, only a per-event count, to avoid an extra ragged collection.

    Adds:
    - `{topocluster_collection}_pred_match_idx` on the topocluster collection: for each node, the
      index of the best-matching predicted particle (as above), -1 if no query predicts this node,
      -2 if the best match is an unmatched/extra query slot (>= n_particles).
    - `{particle_collection}_pred_class`, `_pred_valid`, `_pred_n_nodes`, and `_pred_<field>` for
      each regressed field (plus a derived `_pred_phi` when sinphi/cosphi are regressed), on the
      particle collection. The entrance-position targets come out as `_pred_boundary_x`,
      `_pred_boundary_y`, ... (cm, un-scaled with their own entries of the particle dict).
    - `n_fake_particles` per event, at the top level.

    Also accumulates and reports the mean test loss per layer/task/loss term, since the base
    `ModelWrapper.test_step` does not log losses itself.
    """

    def __init__(self, output_path: str | None = None) -> None:
        super().__init__()
        self._output_path_override = output_path

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return

        self.trainer = trainer
        self.ds = trainer.datamodule.test_dataloader().dataset
        # total SimTracksters per event (valid or not), keyed by event_number: per-particle output fields must have
        # the collection's length even when some particles are not valid targets (require_nodes_for_valid)
        self._n_all_by_event = dict(zip(np.asarray(self.ds.event_number).tolist(), np.asarray(self.ds.n_particles).tolist()))
        self.test_path = trainer.datamodule.test_path
        self.test_suff = trainer.datamodule.test_suff
        self.topocluster_collection = self.ds.topocluster_collection
        self.particle_collection = self.ds.particle_collection
        # regression outputs live in the PARTICLE-fitted space (dataset targets and the
        # task's proxy features both use particle_scaler), so un-scale with that dict
        self.var_transform = self.ds.particle_scaler.transforms

        self._event_numbers = []
        self._node_pred_match_idx = []
        self._particle_pred_class = []
        self._particle_pred_valid = []
        self._particle_pred_n_nodes = []
        self._particle_pred_regr = defaultdict(list)
        self._n_fake_particles = []
        self._particle_pred_class_prob = []   # (n_particles, num_classes+1) per event, null = last column
        self._event_max_valid_prob = []       # max over ALL queries of 1 - p(null)
        self._event_max_valid_prob_class = [] # argmax class of that query
        self._particle_pred_class_prob_l3 = []  # same from the last intermediate decoder layer, if present
        self._n_fake_particles_l3 = []           # valid (argmax != null) unmatched queries per event, layer-3 head
        self._fake_thr = (0.5, 0.3, 0.2, 0.1, 0.05)
        self._n_fake_l3_thr = []                 # per event: [n unmatched with p_null_l3 < t for t in _fake_thr]
        self._n_fake_l3_thr_mask = []            # same, additionally requiring a non-empty predicted mask
        self._n_matched_l3_thr = []              # per event: [n matched (true-particle) queries with p_null_l3 < t]

        self._loss_sums: dict[tuple[str, str, str], float] = {}
        self._loss_counts: dict[tuple[str, str, str], int] = {}

    @property
    def output_path(self) -> Path:
        if self._output_path_override is not None:
            return Path(self._output_path_override)
        ckpt_path = Path(self.trainer.ckpt_path)
        suffix = f"_{self.test_suff}" if self.test_suff else ""
        return ckpt_path.parent / f"{ckpt_path.stem}__test{suffix}.parquet"

    @property
    def loss_output_path(self) -> Path:
        return self.output_path.with_name(f"{self.output_path.stem}_losses.json")

    @staticmethod
    def _to_jagged(per_event_arrays: list[np.ndarray]) -> ak.Array:
        """Build a jagged ak.Array from a list of per-event numpy arrays, preserving dtype.

        `ak.Array(per_event_arrays)` builds through a generic iterable path that promotes
        float32 arrays to float64; concatenate+unflatten instead reuses the input buffers.
        """
        counts = np.array([len(arr) for arr in per_event_arrays])
        dtype = per_event_arrays[0].dtype if per_event_arrays else np.float64
        flat = np.concatenate(per_event_arrays) if per_event_arrays else np.array([], dtype=dtype)
        return ak.unflatten(flat, counts)

    def _unscale_regression_preds(self, preds_final: dict) -> dict[str, torch.Tensor]:
        raw = {
            key[len("pflow_") :]: val.detach().cpu()
            for key, val in preds_final["regression"].items()
            if key.startswith("pflow_") and "proxy" not in key
        }
        unscaled = {}
        for field, val in raw.items():
            unscaled[field] = self.var_transform[field].inverse_transform(val) if field in self.var_transform else val
        if "sinphi" in unscaled and "cosphi" in unscaled:
            unscaled["phi"] = torch.atan2(unscaled["sinphi"], unscaled["cosphi"])
        return unscaled

    def on_test_batch_end(self, trainer, pl_module, test_step_outputs, batch, batch_idx) -> None:
        _inputs, targets = batch
        _outputs, preds, losses = test_step_outputs
        preds_final = preds["final"]

        event_number = targets["event_number"].cpu().numpy().astype(np.int64)
        particle_valid = targets["particle_valid"].cpu()  # (b, num_objects)
        node_valid = targets["node_valid"].cpu()  # (b, max_nodes)

        pred_class = preds_final["classification"]["pflow_class"].detach().cpu()
        class_prob = _outputs["final"]["classification"]["pflow_class_prob"].detach().float().cpu()  # (b, num_objects, C+1)
        valid_prob_all = 1.0 - class_prob[..., -1]  # (b, num_objects)
        inter = [k for k in _outputs if k.startswith("layer_")]
        class_prob_l3 = _outputs[sorted(inter)[-1]]["classification"]["pflow_class_prob"].detach().float().cpu() if inter else None
        pred_obj_valid = preds_final["classification"]["pflow_valid"].detach().cpu()
        mask_valid = preds_final["mask"]["pflow_node_valid"].detach().cpu()  # (b, num_objects, max_nodes)
        incidence = preds_final["incidence"]["pflow_incidence"].detach().cpu()  # (b, num_objects, max_nodes)
        reg_preds = self._unscale_regression_preds(preds_final)

        # Best-matching query per node: highest incidence among queries that predict this node as valid.
        masked_incidence = incidence.masked_fill(~mask_valid, -1.0)
        has_any_query = mask_valid.any(dim=1)  # (b, max_nodes)
        best_query = masked_incidence.argmax(dim=1)  # (b, max_nodes)

        batch_size = event_number.shape[0]
        for i in range(batch_size):
            n_particles = int(self._n_all_by_event.get(int(event_number[i]), int(particle_valid[i].sum())))
            tvalid = particle_valid[i, :n_particles].numpy().astype(bool)   # which of the event's particles are targets
            not_target = ~particle_valid[i].numpy().astype(bool)             # all 150 slots that are not valid targets
            n_nodes = int(node_valid[i].sum())

            bq = best_query[i, :n_nodes]
            match_idx = torch.where(bq < n_particles, bq, torch.full_like(bq, -2))
            match_idx = torch.where(has_any_query[i, :n_nodes], match_idx, torch.full_like(match_idx, -1))
            self._node_pred_match_idx.append(match_idx.numpy().astype(np.int64))

            self._n_fake_particles.append(int(pred_obj_valid[i].numpy()[not_target].sum()))
            pc = pred_class[i, :n_particles].numpy().astype(np.int64); pc[~tvalid] = class_prob.shape[-1] - 1
            self._particle_pred_class.append(pc)
            pv = pred_obj_valid[i, :n_particles].numpy().copy(); pv[~tvalid] = False
            self._particle_pred_valid.append(pv)
            cp = class_prob[i, :n_particles].numpy().astype(np.float32).copy(); cp[~tvalid] = np.nan
            self._particle_pred_class_prob.append(cp)
            if class_prob_l3 is not None:
                cp3 = class_prob_l3[i, :n_particles].numpy().astype(np.float32).copy(); cp3[~tvalid] = np.nan
                self._particle_pred_class_prob_l3.append(cp3)
                valid_l3 = (class_prob_l3[i].argmax(-1) < (class_prob_l3.shape[-1] - 1)).numpy()
                self._n_fake_particles_l3.append(int(valid_l3[not_target].sum()))
                pn3 = class_prob_l3[i, :, -1].numpy(); nonempty = (mask_valid[i].sum(-1) > 0).numpy()
                self._n_fake_l3_thr.append([int((pn3[not_target] < t).sum()) for t in self._fake_thr])
                self._n_fake_l3_thr_mask.append([int(((pn3[not_target] < t) & nonempty[not_target]).sum()) for t in self._fake_thr])
                self._n_matched_l3_thr.append([int((pn3[:n_particles][tvalid] < t).sum()) for t in self._fake_thr])
            j = int(valid_prob_all[i].argmax())
            self._event_max_valid_prob.append(float(valid_prob_all[i, j]))
            self._event_max_valid_prob_class.append(int(class_prob[i, j, :-1].argmax()))
            pnn = mask_valid[i, :n_particles].sum(-1).numpy().astype(np.int64); pnn[~tvalid] = 0
            self._particle_pred_n_nodes.append(pnn)
            for field, val in reg_preds.items():
                rv = val[i, :n_particles].numpy().astype(np.float32).copy(); rv[~tvalid] = np.nan
                self._particle_pred_regr[field].append(rv)

            self._event_numbers.append(int(event_number[i]))

        for layer_name, layer_losses in losses.items():
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    key = (layer_name, task_name, loss_name)
                    self._loss_sums[key] = self._loss_sums.get(key, 0.0) + float(loss_value.detach().cpu()) * batch_size
                    self._loss_counts[key] = self._loss_counts.get(key, 0) + batch_size

    def on_test_end(self, trainer, pl_module) -> None:
        event_numbers = np.array(self._event_numbers, dtype=np.int64)

        # event_numbers index the concatenation of exactly the files the test dataset read
        print(f"Loading original test file(s) to merge in predictions: {self.test_path} ({len(self.ds.files)} files)")
        orig = ak.from_arrow(read_parquet_files(self.ds.files))
        out = orig[event_numbers]

        topo = ak.with_field(
            out[self.topocluster_collection],
            self._to_jagged(self._node_pred_match_idx),
            f"{self.topocluster_collection}_pred_match_idx",
        )
        out = ak.with_field(out, topo, self.topocluster_collection)

        particle = out[self.particle_collection]
        particle = ak.with_field(particle, self._to_jagged(self._particle_pred_class), f"{self.particle_collection}_pred_class")
        particle = ak.with_field(particle, self._to_jagged(self._particle_pred_valid), f"{self.particle_collection}_pred_valid")
        particle = ak.with_field(particle, self._to_jagged(self._particle_pred_n_nodes), f"{self.particle_collection}_pred_n_nodes")
        for field, values in self._particle_pred_regr.items():
            particle = ak.with_field(particle, self._to_jagged(values), f"{self.particle_collection}_pred_{field}")
        out = ak.with_field(out, particle, self.particle_collection)

        out = ak.with_field(out, ak.Array(self._n_fake_particles), "n_fake_particles")
        n_cls = self._particle_pred_class_prob[0].shape[1] if self._particle_pred_class_prob else 0
        names = ["hadronic", "electromagnetic", "muon", "null"] if n_cls == 4 else [f"class{k}" for k in range(n_cls)]
        for k, nm in enumerate(names):
            particle = ak.with_field(particle, self._to_jagged([a[:, k] for a in self._particle_pred_class_prob]), f"{self.particle_collection}_pred_prob_{nm}")
        if self._particle_pred_class_prob_l3:
            for k, nm in enumerate(names):
                particle = ak.with_field(particle, self._to_jagged([a[:, k] for a in self._particle_pred_class_prob_l3]), f"{self.particle_collection}_pred_prob_l3_{nm}")
        out = ak.with_field(out, particle, self.particle_collection)
        out = ak.with_field(out, ak.Array(np.array(self._event_max_valid_prob, dtype=np.float32)), "max_valid_prob_any_query")
        if self._n_fake_particles_l3:
            out = ak.with_field(out, ak.Array(np.array(self._n_fake_particles_l3, dtype=np.int64)), "n_fake_particles_l3")
            for k, t in enumerate(self._fake_thr):
                tag = str(t).replace(".", "p")
                out = ak.with_field(out, ak.Array(np.array([v[k] for v in self._n_fake_l3_thr], dtype=np.int64)), f"n_fake_l3_pnull_lt_{tag}")
                out = ak.with_field(out, ak.Array(np.array([v[k] for v in self._n_fake_l3_thr_mask], dtype=np.int64)), f"n_fake_l3_pnull_lt_{tag}_mask")
                out = ak.with_field(out, ak.Array(np.array([v[k] for v in self._n_matched_l3_thr], dtype=np.int64)), f"n_matched_l3_pnull_lt_{tag}")
        out = ak.with_field(out, ak.Array(np.array(self._event_max_valid_prob_class, dtype=np.int64)), "max_valid_prob_any_query_class")
        out = ak.with_field(out, ak.Array(event_numbers), "event_number")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        ak.to_parquet(out, self.output_path)
        # remembered for whoever runs after the test loop (the plotting chain in
        # evaluate_maskformer.py), independent of trainer.ckpt_path surviving teardown
        self.written_output_path = self.output_path
        print(f"Wrote {len(event_numbers)} events with predictions to {self.output_path}")

        summary: dict = {}
        for (layer_name, task_name, loss_name), total in self._loss_sums.items():
            count = self._loss_counts[(layer_name, task_name, loss_name)]
            summary.setdefault(layer_name, {}).setdefault(task_name, {})[loss_name] = total / count
        summary["total_loss"] = sum(v for layer in summary.values() if isinstance(layer, dict) for task in layer.values() for v in task.values())

        print("Test loss summary:")
        print(json.dumps(summary, indent=2))
        with self.loss_output_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote loss summary to {self.loss_output_path}")
