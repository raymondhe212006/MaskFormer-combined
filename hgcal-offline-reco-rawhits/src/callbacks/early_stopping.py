import os

from lightning.pytorch.callbacks import EarlyStopping as _EarlyStopping


class EarlyStopping(_EarlyStopping):
    """Stops the training when the total validation loss stops improving.

    Thin layer over Lightning's `EarlyStopping` with this project's defaults: the monitored
    metric is `val/loss` (the total validation loss logged by hepattn's `ModelWrapper`) and
    lower is better. After every validation epoch the current `val/loss` is compared with the
    best one seen so far; if it has not decreased (by more than `min_delta`) for `patience`
    consecutive epochs the training ends, whatever `trainer.max_epochs` says. A non-finite
    (NaN/inf) `val/loss` also stops the training (`check_finite`).

    Configure it in the trainer callbacks, e.g.

        - class_path: callbacks.early_stopping.EarlyStopping
          init_args:
            patience: 25

    Remove the entry from the callbacks list to train for the full `max_epochs`.

    Args:
        monitor: metric to watch. Default `val/loss`.
        patience: number of consecutive validation epochs without improvement after which the
            training stops. Default 25.
        mode: `min` (the default) or `max`, whether lower or higher is better.
        min_delta: minimum decrease (or increase, in `max` mode) of the metric that counts as an
            improvement. Default 0, i.e. any decrease counts.
        verbose: log every improvement and the reason for stopping. Default True.
        **kwargs: forwarded to Lightning's `EarlyStopping` (`check_finite`,
            `stopping_threshold`, `divergence_threshold`, ...).
    """

    def __init__(
        self,
        monitor: str = "val/loss",
        patience: int = 25,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        # EARLY_STOP_STRICT=0 (smoke/profiling runs with validation disabled): skip instead of raising when `monitor` is missing
        kwargs.setdefault("strict", os.environ.get("EARLY_STOP_STRICT", "1") != "0")
        super().__init__(monitor=monitor, patience=patience, mode=mode, min_delta=min_delta, verbose=verbose, **kwargs)
