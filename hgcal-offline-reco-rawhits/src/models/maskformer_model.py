from pathlib import Path

import torch
import torchmetrics as tm
from torch import nn

from hepattn.experiments.clic.metrics import MaskInference
from hepattn.models.wrapper import ModelWrapper

from dataset.dataset_maskformers import NUM_CLASSES, RESIDUAL_CLASS
from dataset.finite_checks import assert_batch_finite, assert_losses_finite


class HGCMaskformer(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
        nonfinite_dump_dir: str | None = None,
    ):
        """Args:
        nonfinite_dump_dir: where the batch that produced a non-finite loss is saved before
            the AssertionError is raised (see `_guard_losses`); defaults to the trainer's log
            directory.
        """
        super().__init__(name, model, lrs_config, optimizer, mtl)
        self.MI = MaskInference
        self.nonfinite_dump_dir = nonfinite_dump_dir
        self._current_batch: tuple[str, int, tuple] | None = None

        self.obj_accuracy_micro = tm.classification.MulticlassAccuracy(num_classes=NUM_CLASSES + 1, average="micro")
        self.obj_accuracy_macro = tm.classification.MulticlassAccuracy(num_classes=NUM_CLASSES + 1, average="macro")

        self.eff = tm.classification.BinaryRecall()
        self.fake = tm.classification.BinaryRecall()
        self.pur = tm.classification.BinaryPrecision()

    # ---- guards against non-finite values -------------------------------------------------
    # A NaN in one event makes the epoch-mean validation loss NaN while the on-step training
    # loss can still look fine (and an L1 term with a NaN target even keeps finite gradients),
    # so a bad batch is hard to trace after the fact. Every batch is checked on arrival and
    # every loss term after it is computed; the first offender raises with the loss terms, the
    # event numbers and a dump of the batch for offline replay.

    def on_train_batch_start(self, batch, batch_idx):
        self._guard_batch(batch, "train", batch_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self._guard_batch(batch, "val", batch_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self._guard_batch(batch, "test", batch_idx)

    def _guard_batch(self, batch, stage, batch_idx):
        self._current_batch = (stage, batch_idx, batch)
        inputs, targets = batch
        assert_batch_finite(inputs, targets, context=f"{stage} batch {batch_idx}")

    def aggregate_losses(self, losses, stage=None):
        # hepattn >= 0.2: called by training/validation_step right after model.loss
        self._guard_losses(losses, stage)
        return super().aggregate_losses(losses, stage)

    def log_losses(self, losses, stage):
        # Older hepattn: same role as aggregate_losses under its previous name
        self._guard_losses(losses, stage)
        return super().log_losses(losses, stage)

    def _guard_losses(self, losses, stage):
        event_numbers = None
        if self._current_batch is not None:
            event_numbers = self._current_batch[2][1].get("event_number")
        try:
            assert_losses_finite(losses, stage=stage or "", event_numbers=event_numbers)
        except AssertionError as err:
            dump = self._dump_batch(losses, stage)
            raise AssertionError(f"{err}\n{dump}") from None

    def _dump_batch(self, losses, stage) -> str:
        """Save the current batch (inputs, targets, loss values) for offline replay; returns a
        message saying where, or why it could not be saved. Never raises."""
        if self._current_batch is None:
            return "No batch recorded, nothing dumped."
        _, batch_idx, (inputs, targets) = self._current_batch
        trainer = getattr(self, "_trainer", None)
        log_dir = getattr(trainer, "log_dir", None) if trainer is not None else None
        step = getattr(trainer, "global_step", 0) if trainer is not None else 0
        path = Path(self.nonfinite_dump_dir or log_dir or ".") / f"nonfinite_{stage or 'loss'}_step{step}_batch{batch_idx}.pt"
        try:
            to_cpu = lambda d: {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}  # noqa: E731
            loss_values = {
                layer: {task: {name: float(torch.as_tensor(v).float().mean()) for name, v in task_losses.items()} for task, task_losses in layer_losses.items()}
                for layer, layer_losses in losses.items()
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"inputs": to_cpu(inputs), "targets": to_cpu(targets), "losses": loss_values}, path)
        except Exception as exc:  # noqa: BLE001 - the dump must never hide the assertion
            return f"Could not save the batch to {path}: {exc!r}"
        return f"The batch was saved to {path} (dict with 'inputs', 'targets', 'losses') for offline replay."

    def log_custom_metrics(self, preds, labels, stage):
        kwargs = {"sync_dist": True, "batch_size": 1}

        # skip detailed metrics for efficiency
        if stage == "train":
            return

        # get info
        pred_valid = None
        particle_class_labels = labels["particle_class"].squeeze()
        truth_valid = particle_class_labels < RESIDUAL_CLASS
        for key, value in preds.items():
            for task_name, task_value in value.items():
                if task_name == "classification":
                    particle_class_preds = task_value["pflow_class"].squeeze()  # remove batch index

                    # object class prediction metrics
                    valid_class_mask = (particle_class_labels >= 0) & (particle_class_labels <= RESIDUAL_CLASS)
                    valid_class_mask = valid_class_mask.view(-1)
                    self.obj_accuracy_micro(particle_class_preds.view(-1)[valid_class_mask], particle_class_labels.view(-1)[valid_class_mask])
                    self.log(f"{stage}/{key}_obj_class_accuracy_micro", self.obj_accuracy_micro, **kwargs)
                    self.obj_accuracy_macro(particle_class_preds.view(-1)[valid_class_mask], particle_class_labels.view(-1)[valid_class_mask])
                    self.log(f"{stage}/{key}_obj_class_accuracy_macro", self.obj_accuracy_macro, **kwargs)

                    # tracking efficiency and fake rate
                    pred_valid = particle_class_preds < RESIDUAL_CLASS
                    self.eff(pred_valid, truth_valid)
                    self.pur(pred_valid, truth_valid)

                    self.log(f"{stage}/{key}_pur", self.pur, **kwargs)
                    self.log(f"{stage}/{key}_eff", self.eff, **kwargs)
                if task_name == "mask":
                    pred_masks = task_value["pflow_node_valid"].squeeze()  # remove batch index
                    truth_masks = labels["particle_node_valid"].squeeze()  # remove batch index

                    self.mask_metrics(pred_masks, truth_masks, pred_valid, truth_valid, f"{stage}/{key}", **kwargs)

        # regression metrics
        # if "regression" not in preds:
        #     return

        # valid_idx = ~torch.isnan(labels["regression"]).all(-1)
        # valid_idx &= ~(labels["regression"] == 0).all(-1)
        # reg_idx = valid_idx & pred_valid
        # for i, t in enumerate(self.reg_tgts):
        #     reg_preds = preds["regression"][reg_idx][..., i]
        #     targets = labels["regression"][reg_idx][..., i]

        #     reg_preds = reg_preds
        #     targets = targets

        #     # compute and log absolute error
        #     mae = nn.functional.l1_loss(reg_preds, targets)
        #     self.log(f"{stage}_{t}_mae", mae, **kwargs)

    def mask_metrics(self, pred_masks, truth_masks, pred_valid, truth_valid, prefix, **kwargs):
        # Valid objects should be connected at least to one node
        if pred_valid is not None:
            pred_valid &= pred_masks.sum(-1) >= 1
        else:
            pred_valid = pred_masks.sum(-1) >= 1

        # number of nodes per track
        # self.log(f"{prefix}_nn_per_particle", float(pred_masks.sum(-1).float().mean()), **kwargs)
        # self.log(f"{prefix}_nn_per_validparticle", pred_masks[truth_valid].sum(-1).float().mean(), **kwargs)
        # self.log(f"{prefix}_nn_per_invalidparticle", pred_masks[~truth_valid].sum(-1).float().mean(), **kwargs)

        # mask metrics
        truth_masks = truth_masks[truth_valid]
        pred_masks = pred_masks[truth_valid]

        self.log(f"{prefix}_mask_exact_match", self.MI.exact_match(pred_masks, truth_masks), **kwargs)

        # with the hit filter, can get 0 in the denominator here
        # should use the unfiltered target masks or only select targets with at least one hit
        recall_idx = truth_masks.sum(-1) > 0  # get truth masks with at least one hit
        self.log(f"{prefix}_mask_recall", self.MI.eff(pred_masks[recall_idx], truth_masks[recall_idx]), **kwargs)
        pur_idx = pred_masks.sum(-1) > 0  # get reco masks with at least one predicted hit
        self.log(f"{prefix}_mask_purity", self.MI.pur(pred_masks[pur_idx], truth_masks[pur_idx]), **kwargs)
