"""Tests for the training callbacks in src/callbacks (BestCheckpoint, EarlyStopping) on a toy
LightningModule whose validation loss follows a fixed per-epoch schedule. Needs torch and
lightning (skipped otherwise):

    PYTHONPATH=src python -m pytest tests/test_callbacks.py -q
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")
lightning = pytest.importorskip("lightning")

from callbacks import BestCheckpoint, EarlyStopping  # noqa: E402


class ScheduledLossModel(lightning.LightningModule):
    """Logs `val/loss` = val_losses[current_epoch] (the last value repeats past the end)."""

    def __init__(self, val_losses: list[float]) -> None:
        super().__init__()
        self.val_losses = val_losses
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        return self.layer(x).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        value = self.val_losses[min(self.current_epoch, len(self.val_losses) - 1)]
        self.log("val/loss", torch.tensor(value), sync_dist=False)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _loader() -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(4, 1)), batch_size=2)


def _fit(tmp_path: Path, val_losses: list[float], callbacks: list, max_epochs: int) -> lightning.Trainer:
    trainer = lightning.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    trainer.fit(ScheduledLossModel(val_losses), train_dataloaders=_loader(), val_dataloaders=_loader())
    return trainer


def test_best_checkpoint_keeps_only_the_latest_best(tmp_path):
    # Improves at epochs 0, 1, 3; epochs 2, 4 are worse or equal
    val_losses = [3.0, 2.0, 2.5, 1.25, 1.25]
    _fit(tmp_path, val_losses, [BestCheckpoint()], max_epochs=5)

    ckpts = sorted(p.name for p in (tmp_path / "ckpts").glob("*.ckpt"))
    assert ckpts == ["best-epoch=003-val_loss=1.25000.ckpt"]

    # The file is a full Lightning checkpoint of that epoch
    state = torch.load(tmp_path / "ckpts" / ckpts[0], map_location="cpu", weights_only=False)
    assert state["epoch"] == 3
    assert "state_dict" in state


def test_best_checkpoint_coexists_with_a_per_epoch_checkpoint(tmp_path):
    from lightning.pytorch.callbacks import ModelCheckpoint

    # Stand-in for hepattn's Checkpoint: every epoch, same directory, same monitor
    class Checkpoint(ModelCheckpoint):
        def __init__(self):
            super().__init__(
                save_top_k=-1,
                monitor="val/loss",
                filename="epoch={epoch:03d}-val_loss={val/loss:.5f}",
                auto_insert_metric_name=False,
                dirpath=str(tmp_path / "ckpts"),
            )

    _fit(tmp_path, [2.0, 1.0, 1.5], [Checkpoint(), BestCheckpoint()], max_epochs=3)

    ckpts = sorted(p.name for p in (tmp_path / "ckpts").glob("*.ckpt"))
    assert ckpts == [
        "best-epoch=001-val_loss=1.00000.ckpt",
        "epoch=000-val_loss=2.00000.ckpt",
        "epoch=001-val_loss=1.00000.ckpt",
        "epoch=002-val_loss=1.50000.ckpt",
    ]


def test_early_stopping_stops_after_patience_epochs_without_improvement(tmp_path):
    # Last improvement at epoch 2, patience 3 -> stop after epoch 5 (epochs 3, 4, 5 no better)
    val_losses = [3.0, 2.0, 1.0, 1.0, 1.2, 1.0, 0.5, 0.1]
    early_stopping = EarlyStopping(patience=3)
    trainer = _fit(tmp_path, val_losses, [early_stopping, BestCheckpoint()], max_epochs=20)

    assert early_stopping.stopped_epoch == 5
    assert trainer.current_epoch == 6  # epochs 0..5 ran
    assert float(early_stopping.best_score) == 1.0
    assert [p.name for p in (tmp_path / "ckpts").glob("*.ckpt")] == ["best-epoch=002-val_loss=1.00000.ckpt"]


def test_early_stopping_does_not_stop_while_the_loss_improves(tmp_path):
    val_losses = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
    early_stopping = EarlyStopping(patience=2)
    trainer = _fit(tmp_path, val_losses, [early_stopping], max_epochs=6)

    assert early_stopping.stopped_epoch == 0  # never triggered
    assert trainer.current_epoch == 6


def test_early_stopping_defaults():
    cb = EarlyStopping()
    assert cb.monitor == "val/loss"
    assert cb.patience == 25
    assert cb.mode == "min"
    assert cb.min_delta == 0.0
