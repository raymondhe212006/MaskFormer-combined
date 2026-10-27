"""Evaluation script for a trained MaskFormer model, powered by the lightning CLI.

Loads the best (or a given) checkpoint, runs inference on a test parquet file, computes
predictions and losses, writes the predictions to a parquet file alongside the checkpoint
(same schema as the input file, plus prediction branches - see callbacks.prediction_writer),
and then runs the standard plotting chain (src/scripts/plot/run_plot_chain.py) on that file.

Example usage:

    python src/scripts/evaluate/evaluate_maskformer.py test \\
        --config configs/test.yaml \\
        --data.test_path /path/to/test.parquet \\
        --trainer.precision 32-true \\
        --matmul_precision highest

Omit --ckpt_path to automatically use the best checkpoint (lowest val loss) next to the config.

Plots: by default they are written next to the prediction file, in
`<ckpt_dir>/<ckpt_stem>__test[_<test_suff>]_plots/` (resolution/, resolution_by_class/,
efficiency/, diagnostics/, cluster_properties/, topocluster_features/, event_reconstruction/,
event_display/, energy/, metrics.json, summary.txt). The node input list and the node scale
dict of the data config are passed to the chain for the topocluster_features plots. Options:

    --plot_dir <dir>          write the plots somewhere else
    --skip_plots true         only write the prediction file
    --plot_max_events <N>     plot the first N events only (smoke test)
    --plot_n_display <N>      number of event displays (default 5, 0 disables)

The chain runs as a subprocess of the same interpreter (override with the environment
variable HGCAL_PLOT_PYTHON); a plotting failure never fails the evaluation, the command to
rerun the chain by hand is printed instead.
"""

import gc
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

import comet_ml  # noqa: F401
from lightning.pytorch.cli import ArgsType

from callbacks.prediction_writer import HGCPredictionWriter
from models.maskformer_model import HGCMaskformer
from dataset.dataset_maskformers import HGCDataModule
from hepattn.utils.cli import CLI

PLOT_CHAIN = Path(__file__).resolve().parents[1] / "plot" / "run_plot_chain.py"


class EvalCLI(CLI):
    """hepattn CLI plus the plotting chain after `test`."""

    def add_arguments_to_parser(self, parser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--plot_dir",
            type=Optional[str],
            default=None,
            help="Directory for the evaluation plots. Default: next to the prediction file, "
            "<ckpt_dir>/<ckpt_stem>__test[_<test_suff>]_plots/.",
        )
        parser.add_argument(
            "--skip_plots",
            type=bool,
            default=False,
            help="Do not run the plotting chain after the test loop (`--skip_plots true`).",
        )
        parser.add_argument(
            "--plot_max_events",
            type=Optional[int],
            default=None,
            help="Plot only the first N events of the prediction file (smoke test; the file is "
            "still read in full).",
        )
        parser.add_argument(
            "--plot_n_display",
            type=int,
            default=5,
            help="Number of events drawn in event_display/ (0 disables the displays).",
        )

    # LightningCLI calls `after_<subcommand>` once the subcommand has returned
    def after_test(self) -> None:
        sc = self.config[self.subcommand]
        if sc.get("skip_plots"):
            print("[plots] --skip_plots set: not running the plotting chain")
            return
        if not self.trainer.is_global_zero:
            return
        try:
            self._run_plot_chain(sc)
        except Exception:  # noqa: BLE001 - the prediction file is written, never fail here
            traceback.print_exc()
            print("[plots] plotting failed (see traceback above); the prediction file is unaffected")

    def _run_plot_chain(self, sc) -> None:
        writer = next((c for c in self.trainer.callbacks if isinstance(c, HGCPredictionWriter)), None)
        if writer is None:
            print("[plots] no HGCPredictionWriter among trainer.callbacks: nothing to plot")
            return
        parquet = Path(getattr(writer, "written_output_path", None) or writer.output_path)
        if not parquet.is_file():
            print(f"[plots] prediction file {parquet} not found: nothing to plot")
            return

        plot_dir = Path(sc["plot_dir"]) if sc.get("plot_dir") else parquet.with_name(parquet.stem + "_plots")

        # collection choices of the model, so the chain does not have to guess them from the file
        ds = getattr(writer, "ds", None)
        topocluster_collection = writer.topocluster_collection
        particle_collection = writer.particle_collection
        association_collection = getattr(ds, "association_collection", None) if getattr(ds, "has_soft_links", False) else None
        min_link_fraction = getattr(ds, "min_link_fraction", 0.0)
        # node input features and their scale dict (data.inputs.topocluster, data.scale_dict_path),
        # for the raw / transformed node-feature plots
        datamodule = self.trainer.datamodule
        inputs = getattr(ds, "inputs", None) or (getattr(datamodule, "kwargs", None) or {}).get("inputs") or {}
        node_features = [str(f) for f in (inputs.get("topocluster") or [])]
        scale_dict = getattr(datamodule, "scale_dict_path", None)
        scale_dict = str(Path(scale_dict).resolve()) if scale_dict and Path(scale_dict).is_file() else None

        # the test dataset can be several GB (rechits); release it before spawning the chain
        writer.ds = None
        if datamodule is not None and hasattr(datamodule, "test_dset"):
            datamodule.test_dset = None
        gc.collect()

        cmd = [
            os.environ.get("HGCAL_PLOT_PYTHON", sys.executable),
            str(PLOT_CHAIN),
            "--input", str(parquet),
            "--outdir", str(plot_dir),
            "--topocluster-collection", topocluster_collection,
            "--particle-collection", particle_collection,
        ]
        if association_collection:
            cmd += ["--association-collection", association_collection, "--min-link-fraction", str(min_link_fraction)]
        if sc.get("plot_max_events") is not None:
            cmd += ["--max-events", str(sc["plot_max_events"])]
        if scale_dict:
            cmd += ["--scale-dict", scale_dict]
        if node_features:
            cmd += ["--node-features", *node_features]
        if sc.get("plot_n_display") is not None:
            cmd += ["--n-display", str(sc["plot_n_display"])]

        print("-" * 100)
        print("[plots] running the plotting chain:\n  " + shlex.join(cmd), flush=True)
        rc = subprocess.run(cmd, check=False).returncode  # noqa: S603
        if rc == 0:
            print(f"[plots] done -> {plot_dir}")
        else:
            print(f"[plots] plot chain exited with code {rc}; rerun it by hand with\n  {shlex.join(cmd)}")
        print("-" * 100)


def main(args: ArgsType = None) -> None:
    EvalCLI(
        model_class=HGCMaskformer,
        datamodule_class=HGCDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
