from pathlib import Path

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


class BestCheckpoint(ModelCheckpoint):
    """Keeps a single copy of the best checkpoint of the run in the `ckpts/` directory.

    hepattn's `Checkpoint` callback writes every epoch as `epoch=NNN-val_loss=X.XXXXX.ckpt`
    into `<log_dir>/ckpts/`. This callback writes, next to those, the checkpoint of the epoch
    with the best monitored metric so far as

        best-epoch=NNN-val_loss=X.XXXXX.ckpt

    and, whenever a later epoch improves on it, removes the previous `best-*.ckpt` and writes
    the new one (`save_top_k=1`, so there is always exactly one `best-*.ckpt`). It does not
    upload the file to the logger (the per-epoch `Checkpoint` already does that).

    `get_best_epoch` in hepattn's CLI (used by `evaluate_maskformer.py` when `--ckpt_path` is
    omitted) reads the loss off the file name, so the `best-*` file is picked up as the best
    checkpoint like any other.

    Args:
        monitor: metric to compare epochs by. Default `val/loss`, the total validation loss.
        mode: `min` (the default) or `max`, whether lower or higher is better.
    """

    def __init__(self, monitor: str = "val/loss", mode: str = "min", **kwargs) -> None:
        filename = "best-epoch={epoch:03d}-" + monitor.replace("/", "_") + "={" + monitor + ":.5f}"
        super().__init__(
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_last=False,
            filename=filename,
            auto_insert_metric_name=False,
            **kwargs,
        )

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
        if stage == "fit" and not trainer.fast_dev_run:
            # Same directory as hepattn's Checkpoint: <log_dir>/ckpts
            self.dirpath = str(Path(trainer.log_dir) / "ckpts")
