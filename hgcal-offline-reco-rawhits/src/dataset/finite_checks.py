"""Assertions that keep NaN/inf out of the training pipeline.

A non-finite value that enters through a raw branch, a scale-dict transform, the incidence
matrix or a model output only surfaces much later as a NaN loss, and a NaN in a single event
makes the validation epoch mean NaN while the on-step training loss can still look fine.
Every check here raises AssertionError at the point of entry, naming the offending quantity
(branch, feature, incidence matrix, loss term) and, where known, the event.

Shared by the dataset (per-file and per-event checks) and the model wrapper (per-batch and
per-loss checks); needs only numpy, scipy and torch.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from scipy.sparse import spmatrix

# Relative tolerance on the entries of the incidence matrix and on its sums. The soft (energy
# share) incidence divides two float32 branches, sharedEnergy and raw_energy, so a trackster
# fully owned by one SimTrackster carries a share of 1 + 8e-7 (measured on
# taus_fineCalo_split.parquet); anything beyond this tolerance is a genuine inconsistency.
INCIDENCE_TOL = 1e-4


def _context(context: str) -> str:
    return f" ({context})" if context else ""


def _nonfinite_count(values) -> tuple[int, int]:
    """(number of non-finite entries, number of entries) of a tensor or array-like."""
    if isinstance(values, torch.Tensor):
        values = values.detach()
        if not values.is_floating_point():
            return 0, values.numel()
        return int((~torch.isfinite(values)).sum()), values.numel()
    arr = np.asarray(values)
    if arr.dtype.kind not in "fc":
        return 0, arr.size
    return int((~np.isfinite(arr)).sum()), arr.size


def assert_finite(name: str, values, context: str = "") -> None:
    """Raise AssertionError if `values` (torch.Tensor or array-like) has NaN or inf entries.

    Integer and boolean inputs pass trivially.
    """
    n_bad, n = _nonfinite_count(values)
    if n_bad:
        raise AssertionError(f"{name}: {n_bad}/{n} non-finite (NaN/inf) entries{_context(context)}")


def assert_scaler_finite(scaler, scale_dict_path: str) -> None:
    """Every transform of a hepattn FeatureScaler must have a finite shift and a finite,
    non-zero scale: a bad entry in the scale dict would turn the whole feature into NaN/inf."""
    for name, transform in list(scaler.transforms.items()):
        shift, scale = float(transform.shift), float(transform.scale)
        if not (np.isfinite(shift) and np.isfinite(scale)) or scale == 0:
            raise AssertionError(
                f"scale dict {scale_dict_path}: variable '{name}' has shift={shift} scale={scale} "
                f"(must be finite with a non-zero scale)"
            )


def assert_transform_domain(name: str, values: torch.Tensor, transform, context: str = "") -> None:
    """The values must lie in the domain of the transform's `fn` (log: > 0, log1p: > -1,
    sqrt: >= 0); otherwise the scaled feature is -inf/NaN for those entries."""
    fn = getattr(transform, "fn", None)
    if fn == "log":
        bad, domain = values <= 0, "> 0"
    elif fn == "log1p":
        bad, domain = values <= -1, "> -1"
    elif fn == "sqrt":
        bad, domain = values < 0, ">= 0"
    else:
        return
    n_bad = int(bad.sum())
    if n_bad:
        raise AssertionError(
            f"{name}: {n_bad}/{values.numel()} values outside the domain {domain} of the '{fn}' transform "
            f"of its scale dict (min {float(values.min()):.4g}){_context(context)}"
        )


def assert_incidence_matrix(incidence, *, row_normalized: bool, context: str = "", tol: float = INCIDENCE_TOL) -> None:
    """Check that an incidence matrix (particles x nodes) is well defined.

    Accepts a scipy sparse matrix, a numpy array or a torch tensor (optionally batched,
    (batch, particles, nodes)). Every entry must be finite and within [0, 1 + tol]. With
    `row_normalized=False` the entries are energy shares (or a one-hot assignment), so the
    shares of one node summed over the particles must not exceed 1 + tol; with
    `row_normalized=True` each particle row was L1-normalised, so it must sum to at most 1 + tol.
    """
    if isinstance(incidence, spmatrix):
        data = np.asarray(incidence.data, dtype=np.float64)
        row_sums = np.asarray(incidence.sum(axis=1)).ravel()
        col_sums = np.asarray(incidence.sum(axis=0)).ravel()
        n_entries = incidence.shape[0] * incidence.shape[1]
    elif isinstance(incidence, torch.Tensor):
        dense = incidence.detach().float()
        data, row_sums, col_sums = dense, dense.sum(-1), dense.sum(-2)
        n_entries = dense.numel()
    else:
        dense = np.asarray(incidence, dtype=np.float64)
        data, row_sums, col_sums = dense, dense.sum(-1), dense.sum(-2)
        n_entries = dense.size

    n_bad, _ = _nonfinite_count(data)
    if n_bad:
        raise AssertionError(f"incidence matrix: {n_bad}/{n_entries} non-finite entries{_context(context)}")
    if data.size if isinstance(data, np.ndarray) else data.numel():
        lo, hi = float(data.min()), float(data.max())
        if lo < 0 or hi > 1 + tol:
            raise AssertionError(f"incidence matrix: entries outside [0, 1] (min {lo:.6g}, max {hi:.6g}){_context(context)}")
    sums, what = (row_sums, "a particle row (L1-normalised rows)") if row_normalized else (col_sums, "a node column (energy shares over the particles)")
    if (sums.size if isinstance(sums, np.ndarray) else sums.numel()) and float(sums.max()) > 1 + tol:
        raise AssertionError(f"incidence matrix: {what} sums to {float(sums.max()):.6g} > 1{_context(context)}")


def _event_list(event_numbers, idx: torch.Tensor, limit: int = 10) -> str:
    """Format the events flagged by the boolean per-event mask `idx`."""
    flagged = idx.nonzero().flatten().tolist()
    if event_numbers is not None:
        flagged = torch.as_tensor(event_numbers).flatten()[flagged].tolist()
        label = "event numbers"
    else:
        label = "batch positions"
    shown = ", ".join(str(e) for e in flagged[:limit]) + (", ..." if len(flagged) > limit else "")
    return f"{len(flagged)} events ({label}: {shown})"


def assert_batch_finite(inputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], context: str = "") -> None:
    """Check one collated batch, as handed to the model, for non-finite values.

    Node inputs are checked on the valid nodes, particle targets on the valid particles (the
    padded regression targets are NaN by construction) and the dense incidence target on every
    entry and its range. The message lists the offending keys and the event numbers.
    """
    node_valid = targets["node_valid"].bool()
    particle_valid = targets["particle_valid"].bool()
    event_numbers = targets.get("event_number")
    batch_size = node_valid.shape[0]
    problems = []

    def check(name: str, tensor: torch.Tensor, valid: torch.Tensor | None) -> None:
        if not (isinstance(tensor, torch.Tensor) and tensor.is_floating_point()):
            return
        bad = ~torch.isfinite(tensor.detach())
        if bad.dim() > 2:
            bad = bad.flatten(2).any(-1)
        if valid is not None:
            bad = bad & valid
        if bad.any():
            problems.append(f"'{name}' non-finite for {_event_list(event_numbers, bad.flatten(1).any(-1))}")

    for key, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2 and tensor.shape[:2] == node_valid.shape:
            check(f"input {key}", tensor, node_valid)
    for key, tensor in targets.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        if key == "particle_incidence":
            bad = ~torch.isfinite(tensor.detach())
            if bad.any():
                problems.append(f"'{key}' non-finite for {_event_list(event_numbers, bad.flatten(1).any(-1))}")
            else:
                out = (tensor < 0) | (tensor > 1 + INCIDENCE_TOL)
                if out.any():
                    problems.append(f"'{key}' outside [0, 1] for {_event_list(event_numbers, out.flatten(1).any(-1))}")
        elif tensor.dim() >= 2 and tensor.shape[:2] == particle_valid.shape:
            check(f"target {key}", tensor, particle_valid)
        elif tensor.dim() >= 2 and tensor.shape[:2] == node_valid.shape:
            check(f"target {key}", tensor, node_valid)
    if problems:
        raise AssertionError(f"non-finite batch of {batch_size} events{_context(context)}:\n  " + "\n  ".join(problems))


def assert_losses_finite(losses: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]], stage: str = "", event_numbers=None) -> None:
    """Raise AssertionError naming every non-finite loss term of hepattn's nested
    {layer: {task: {loss: value}}} dictionary."""
    bad = {}
    for layer, layer_losses in losses.items():
        for task, task_losses in layer_losses.items():
            for name, value in task_losses.items():
                value_t = torch.as_tensor(value)
                if not bool(torch.isfinite(value_t).all()):
                    bad[f"{layer}/{task}/{name}"] = float(value_t) if value_t.numel() == 1 else "non-finite"
    if bad:
        events = ""
        if event_numbers is not None:
            events = f"; batch event numbers: {torch.as_tensor(event_numbers).flatten().tolist()}"
        raise AssertionError(f"non-finite {stage or ''} loss terms: {bad}{events}".replace("non-finite  loss", "non-finite loss"))
