#!/usr/bin/env python3
"""Compute normalization parameters for the maskformer inputs and regression targets.

Two YAML files are written per (topocluster, particle) collection pair:

  --output           fitted on the TOPOCLUSTER collection (RecHitHGC, LayerCluster or
                     ticlTrackstersCLUE3DHigh); scales the node input features in HGCDataset.
  --particle-output  fitted on the PARTICLE collection (MergedSimCluster, SimCluster or
                     ticlSimTracksters); scales the regression targets in HGCDataset AND is
                     the dict the regression task / prediction writer must be given.

Both dicts use the keys e/eta/r, for populations whose spreads differ by orders of
magnitude (a rechit carries ~0.01 GeV, a SimCluster ~10 GeV), which is why they cannot
be one file. See docs/NORMALIZATION_FIX.md.

The output YAML is consumed by ``hepattn.utils.scaling.FeatureScaler`` /
``VarTransform``, which normalizes a variable as::

    x -> fn(x)                      # fn in {None, sqrt, log, log1p}
    x -> (x - shift) / scale        # shift/scale derived from `type`

with

    type=std          -> shift = mean,        scale = std
    type=min_max      -> shift = min,         scale = max - min
    type=min_max_sym  -> shift = (max+min)/2, scale = (max-min)/2

The transform is applied BEFORE the shift/scale, so the ``mean``/``std``/``min``/``max``
written here must be statistics of the *transformed* variable. Getting that backwards is
exactly the bug this script used to have: it accumulated statistics on the raw rechit
energy and then labelled the entry ``fn: sqrt``, which gave
``shift=(115.449+0.004)/2=57.73`` and collapsed every energy input to ~-0.998, i.e. a
constant feature.

Workflow, per variable: derive -> select -> transform -> accumulate statistics -> fit
shift/scale -> check for outliers -> write YAML (+ before/after plots).

Resource note: the gateway node kills processes above 60% CPU, so this script is
single-threaded by default and supports ``--max-files``/``--max-events`` for cheap smoke
tests. Run the full dataset through Slurm (``sbatch jobs/norm_par_job.sh``).
"""

from __future__ import annotations

import os
import sys


def _prescan_threads(default: int = 1) -> int:
    """Read --threads out of argv before numpy/pyarrow are imported.

    The BLAS/OpenMP thread counts are only read at import time, so they have to be set
    before the imports below - argparse runs far too late.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--threads" and i + 1 < len(sys.argv):
            return max(1, int(sys.argv[i + 1]))
        if arg.startswith("--threads="):
            return max(1, int(arg.split("=", 1)[1]))
    return default


_THREADS = _prescan_threads()
for _envvar in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_envvar, str(_THREADS))

import argparse  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass, field, replace  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from glob import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import awkward as ak  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402

# ---------------------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------------------
# These MUST mirror VarTransform.transform in hepattn/src/hepattn/utils/scaling.py. If the
# two ever disagree, the parameters written here describe a different quantity from the one
# the model actually sees, which is silent and hard to debug.
TRANSFORMS: dict[str | None, tuple[Callable[[np.ndarray], np.ndarray], str | None]] = {
    None: (lambda a: a, None),
    "sqrt": (np.sqrt, "nonneg"),  # requires x >= 0
    "log": (np.log, "positive"),  # requires x >  0
    "log1p": (np.log1p, "nonneg"),  # requires x >= 0
}

# Quantile levels reported for every variable.
REPORT_QUANTILES = (0.1, 1.0, 50.0, 99.0, 99.9)

# ---------------------------------------------------------------------------------------
# Outlier thresholds
# ---------------------------------------------------------------------------------------
# n_sigma of an extreme = |extreme - mean| / std, computed on the TRANSFORMED values.
# Exceeding the warn threshold prints a warning; exceeding the fail threshold fails the
# script (unless --allow-outliers). Per-variable fail thresholds are what keep the job
# resilient against known, physical long tails while still catching a genuinely broken
# variable.
NSIGMA_WARN_DEFAULT = 5.0
NSIGMA_FAIL_DEFAULT = 50.0
NSIGMA_FAIL: dict[str, float] = {
    "x": 10.0,
    "y": 10.0,
    "z": 10.0,
    "rho": 20.0,
    "r": 20.0,
    "eta": 10.0,
    # log1p(rechit energy) has a real 4-decade tail: measured ~162 sigma on 42M hits,
    # driven by a handful of >50 GeV hits. Not corruption, so the threshold is generous.
    "e": 250.0,
    # Valid rechit times reach ~34 ns with a 0.42 ns spread: measured 53 sigma on the
    # 100tau sample (11 sigma on the tau20to200 sample). Physical, and harmless for the
    # fit since `t` uses quantile bounds, so the tail never enters shift/scale.
    "t": 75.0,
    "t_filt": 75.0,
    "sinphi": 10.0,
    "cosphi": 10.0,
    # Entrance point of the seed particle into HGCAL (particle dicts only): x/y are bounded by
    # the detector envelope; z sits at the front face (|z| = 321 cm) for most particles with a
    # long tail of secondaries entering deeper (~11 sigma on the fine-calo tau sample).
    "boundary_x": 10.0,
    "boundary_y": 10.0,
    "boundary_z": 40.0,
    "boundary_eta": 10.0,
}


# ---------------------------------------------------------------------------------------
# Variable specification
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VarSpec:
    """How one output variable is derived, transformed and fitted.

    name    : key written to the YAML (and looked up by FeatureScaler)
    source  : key into the dict of derived arrays
    fn      : transform applied before the statistics are accumulated
    type    : shift/scale convention (see module docstring)
    bounds  : "quantile" -> min/max are the clip quantiles (robust against outliers)
              "extremes" -> min/max are the true extremes
    select  : optional mask on the RAW values, applied before `fn`
    emit    : "fitted"   -> write the measured parameters
              "identity" -> write mean=0/std=1 (measure and plot, but do not scale)
    """

    name: str
    source: str
    fn: str | None = None
    type: str = "min_max_sym"
    bounds: str = "quantile"
    select: Callable[[np.ndarray], np.ndarray] | None = None
    select_desc: str = ""
    emit: str = "fitted"
    alias_of: str | None = None  # copy another variable's entry verbatim
    comment: str = ""


def _valid_time(raw: np.ndarray) -> np.ndarray:
    """Rechit time uses -99 as "no valid time"; it is ~99.4% of hits."""
    return raw > -1.0


TOPO_VAR_SPECS: tuple[VarSpec, ...] = (
    # depend on the tail of whichever sample was used).
    VarSpec("y", "y", bounds="extremes", comment="detector envelope"),
    VarSpec("z", "z", bounds="extremes", comment="detector envelope"),
    VarSpec("rho", "rho", bounds="extremes", comment="detector envelope"),
    VarSpec("r", "r", bounds="extremes", comment="detector envelope"),
    VarSpec("eta", "eta", bounds="extremes", comment="physically bounded at +-3.14"),
    # Energy spans ~4 decades, so it is transformed and standardized rather than
    # min-max'd. `log`, not `log1p`: the rechit median is ~0.006 GeV and the layer-cluster
    # median ~0.05 GeV, and log1p(x) ~= x below 1, so log1p leaves the bulk untouched
    # (measured on 100tau rechits: skew raw 41, sqrt 11, log1p 24, log 1.7). Topocluster
    # energies are strictly positive (detector thresholds); apply_transform() refuses to
    # write the file if that ever stops being true. The alternatives table printed by the
    # script shows what each candidate transform would give.
    VarSpec("e", "e", fn="log", type="std", bounds="quantile", comment="4-decade dynamic range, median << 1 GeV"),
    # Electromagnetic (CE-E) part of the trackster raw energy, an optional input. Exactly 0
    # for ~23% of the tracksters (fine-calo tau sample), so log1p instead of log. Tracksters
    # only - see topo_var_specs().
    VarSpec("e_em", "e_em", fn="log1p", type="std", bounds="quantile", comment="can be exactly 0 -> log1p"),
    # Time: statistics over valid hits only, otherwise the -99 sentinel dominates and
    # every hit maps to ~-1. RecHitHGC only - see topo_var_specs(). Files without any valid
    # time (the toy calorimeter writes -99 for every hit) need --identity-time (= --identity t),
    # which emits the identity instead: HGCDataset transforms RecHitHGC_time with `t` even when
    # time is not an input, so the entry must exist and be finite.
    VarSpec("t", "t", bounds="quantile", select=_valid_time, select_desc="t > -1", comment="excludes -99 sentinel"),
    # Kept only so existing configs referring to t_filt keep working. Copied from `t`
    # rather than measured again, so the two entries cannot drift apart.
    VarSpec("t_filt", "t", bounds="quantile", alias_of="t", comment="alias of t (back-compat)"),
    # Already bounded in [-1, 1] and deliberately unscaled. The entries exist so that
    # FeatureScaler(strict=True) finds them instead of falling back silently.
    VarSpec("sinphi", "sinphi", type="std", emit="identity", comment="already in [-1, 1]"),
    VarSpec("cosphi", "cosphi", type="std", emit="identity", comment="already in [-1, 1]"),
)

# Variables that exist only for collections with a time branch (rechits, tracksters; layer
# clusters carry no time).
TIME_VARS = {"t", "t_filt"}
# Variables that exist only for collections with an electromagnetic-energy branch (tracksters).
EM_ENERGY_VARS = {"e_em"}


def topo_var_specs(collection: str) -> tuple[VarSpec, ...]:
    fields = TOPO_FIELD_MAP[collection]
    drop = set()
    if "time" not in fields:
        drop |= TIME_VARS
    if "em_energy" not in fields:
        drop |= EM_ENERGY_VARS
    return tuple(spec for spec in TOPO_VAR_SPECS if spec.name not in drop)


# Regression targets. Keys match the `fields` of the regression task and the particle_data
# dict in HGCDataset.load_event.
PARTICLE_VAR_SPECS: tuple[VarSpec, ...] = (
    # recEnergy can be exactly 0 (SimClusters with no reconstructed energy), and the
    # regression task feeds a zero energy proxy for absent particles, so log would be
    # fatal here; the particle median is ~1.6 GeV, where log1p works well (skew 0.64).
    VarSpec("e", "e", fn="log1p", type="std", bounds="quantile", comment="can be exactly 0 -> log1p"),
    # SimCluster impact points reach |eta| ~ 5.9 (outside the HGCAL acceptance) while
    # the bulk sits within +-3.1; quantile bounds keep the bulk from being squeezed
    # into the middle fifth of [-1, 1] by a handful of out-of-acceptance clusters.
    VarSpec("eta", "eta", bounds="quantile", comment="tails beyond acceptance"),
    VarSpec("r", "r", bounds="quantile", comment="impact-point radius"),
    VarSpec("sinphi", "sinphi", type="std", emit="identity", comment="already in [-1, 1]"),
    VarSpec("cosphi", "cosphi", type="std", emit="identity", comment="already in [-1, 1]"),
    # Entrance-position targets (EXTRA_TARGETS in dataset_maskformers.py): the point where
    # the seed particle crossed the HGCAL front face. Only for collections whose
    # PARTICLE_FIELD_MAP entry has the boundary fields - see particle_var_specs().
    VarSpec("boundary_x", "boundary_x", bounds="quantile", comment="HGCAL entry point, |x| < ~250 cm"),
    VarSpec("boundary_y", "boundary_y", bounds="quantile", comment="HGCAL entry point, |y| < ~250 cm"),
    VarSpec("boundary_z", "boundary_z", bounds="quantile", comment="front face |z| = 321 cm, secondaries deeper"),
    VarSpec("boundary_eta", "boundary_eta", bounds="quantile", comment="eta of the entry point"),
    VarSpec("boundary_sinphi", "boundary_sinphi", type="std", emit="identity", comment="already in [-1, 1]"),
    VarSpec("boundary_cosphi", "boundary_cosphi", type="std", emit="identity", comment="already in [-1, 1]"),
)

# Variables that exist only for particle collections with a boundary position.
BOUNDARY_VARS = {"boundary_x", "boundary_y", "boundary_z", "boundary_eta", "boundary_sinphi", "boundary_cosphi"}
BOUNDARY_FIELDS = ("boundary_x", "boundary_y", "boundary_z")


def particle_var_specs(collection: str) -> tuple[VarSpec, ...]:
    fields = PARTICLE_FIELD_MAP[collection]
    if all(f in fields for f in BOUNDARY_FIELDS):
        return PARTICLE_VAR_SPECS
    return tuple(spec for spec in PARTICLE_VAR_SPECS if spec.name not in BOUNDARY_VARS)

# Raw branch names per topocluster collection. Mirrors TOPO_FIELD_MAP in
# src/dataset/dataset_maskformers.py: fields listed here are read from the file, anything
# else (eta/phi for rechits, R/sinphi/cosphi always) is derived exactly as the dataset does.
TOPO_FIELD_MAP = {
    "RecHitHGC": {"x": "RecHitHGC_x", "y": "RecHitHGC_y", "z": "RecHitHGC_z", "energy": "RecHitHGC_energy", "time": "RecHitHGC_time"},
    "LayerCluster": {
        "x": "LayerCluster_x",
        "y": "LayerCluster_y",
        "z": "LayerCluster_z",
        "eta": "LayerCluster_eta",
        "phi": "LayerCluster_phi",
        "energy": "LayerCluster_energy",
    },
    # TICLv5 CLUE3D tracksters: barycenter position, raw (layer-cluster sum) energy - NOT the
    # TICL DNN `regressed_energy`, the model must not depend on it - its electromagnetic part
    # `raw_em_energy` (optional input `e_em`), and the HGCAL time with the same -99 "no valid
    # time" sentinel as rechits.
    "ticlTrackstersCLUE3DHigh": {
        "x": "ticlTrackstersCLUE3DHigh_barycenter_x",
        "y": "ticlTrackstersCLUE3DHigh_barycenter_y",
        "z": "ticlTrackstersCLUE3DHigh_barycenter_z",
        "eta": "ticlTrackstersCLUE3DHigh_barycenter_eta",
        "phi": "ticlTrackstersCLUE3DHigh_barycenter_phi",
        "energy": "ticlTrackstersCLUE3DHigh_raw_energy",
        "em_energy": "ticlTrackstersCLUE3DHigh_raw_em_energy",
        "time": "ticlTrackstersCLUE3DHigh_time",
    },
}

# Raw branch names per particle collection, for the quantities the regression targets are
# built from. Mirrors PARTICLE_FIELD_MAP in src/dataset/dataset_maskformers.py (the
# pdgId/eventId/bunchCrossing entries are not needed for the fit).
PARTICLE_FIELD_MAP = {
    **{
        c: {
            "e": f"{c}_recEnergy",
            "eta": f"{c}_impactPoint_eta",
            "phi": f"{c}_impactPoint_phi",
            "x": f"{c}_impactPoint_x",
            "y": f"{c}_impactPoint_y",
            "z": f"{c}_impactPoint_z",
            # the impact point is the entry point into HGCAL
            "boundary_x": f"{c}_impactPoint_x",
            "boundary_y": f"{c}_impactPoint_y",
            "boundary_z": f"{c}_impactPoint_z",
            "boundary_eta": f"{c}_impactPoint_eta",
            "boundary_phi": f"{c}_impactPoint_phi",
        }
        for c in ("MergedSimCluster", "SimCluster")
    },
    # TICLv5 SimTracksters: barycenter position and `regressed_energy`, the seed's true
    # energy at the calorimeter boundary (the best achievable energy target; the
    # layer-cluster sum `raw_energy` is not used), plus the seed's entry point into HGCAL
    # (`boundary*`), the target of the entrance-position regression.
    "ticlSimTracksters": {
        "e": "ticlSimTracksters_regressed_energy",
        "eta": "ticlSimTracksters_barycenter_eta",
        "phi": "ticlSimTracksters_barycenter_phi",
        "x": "ticlSimTracksters_barycenter_x",
        "y": "ticlSimTracksters_barycenter_y",
        "z": "ticlSimTracksters_barycenter_z",
        "boundary_x": "ticlSimTracksters_boundaryX",
        "boundary_y": "ticlSimTracksters_boundaryY",
        "boundary_z": "ticlSimTracksters_boundaryZ",
        "boundary_eta": "ticlSimTracksters_boundaryEta",
        "boundary_phi": "ticlSimTracksters_boundaryPhi",
    },
}
PARTICLE_COLLECTIONS = tuple(PARTICLE_FIELD_MAP)

# Default output file names, per collection.
DEFAULT_OUTPUT = {
    "RecHitHGC": "hgcal_rechit_var_transform.yaml",
    "LayerCluster": "hgcal_layercluster_var_transform.yaml",
    "ticlTrackstersCLUE3DHigh": "hgcal_trackster_clue3dhigh_var_transform.yaml",
    "MergedSimCluster": "hgcal_mergedsimcluster_var_transform.yaml",
    "SimCluster": "hgcal_simcluster_var_transform.yaml",
    "ticlSimTracksters": "hgcal_simtrackster_var_transform.yaml",
}


# ---------------------------------------------------------------------------------------
# Streaming statistics
# ---------------------------------------------------------------------------------------
def _combine(n_a: int, mean_a: float, m2_a: float, batch: np.ndarray) -> tuple[float, float]:
    """Fold `batch` into a running (mean, M2) with the Chan et al. parallel formula.

        M2    = sum_i (x_i - mean)^2
        M2_ab = M2_a + M2_b + delta^2 * n_a * n_b / n_ab,   delta = mean_b - mean_a

    M2 is carried instead of a running std because it is the piece that composes exactly
    across batches. `delta` and the weights use the OLD n_a/mean_a, hence the pure function.
    """
    n_b = batch.size
    mean_b = float(batch.mean())
    m2_b = float(((batch - mean_b) ** 2).sum())
    n_ab = n_a + n_b
    delta = mean_b - mean_a
    mean_ab = (n_a * mean_a + n_b * mean_b) / n_ab
    return mean_ab, m2_a + m2_b + delta**2 * n_a * n_b / n_ab


@dataclass
class RunningStats:
    """Exact streaming mean/std/min/max, plus a bounded uniform subsample for quantiles.

    The mean/variance are combined across batches with the Chan et al. parallel formula:

        M2 = sum_i (x_i - mean)^2
        M2_ab = M2_a + M2_b + delta^2 * n_a * n_b / n_ab,  delta = mean_b - mean_a

    which is why `M2` is carried around instead of a running std: it is the only piece
    that composes exactly across batches. `std` is the population std (ddof=0).
    (`hepattn.utils.stats.combine_mean_std` is the same algorithm in vectorized form; this
    script stays free of hepattn imports so it can run in a plain data-prep environment.)

    The subsample is a Bernoulli sample with probability `p`, starting at 1.0. Whenever the
    buffer exceeds `cap` it is thinned by a random half and `p` is halved at the same
    time, so every value ever seen keeps an equal inclusion probability while memory stays
    bounded. Starting from p=1 means small inputs (and rare selections such as the valid
    rechit times) are sampled exactly, and only genuinely large populations get thinned -
    which is why there is no fixed sampling fraction to tune.
    """

    rng: np.random.Generator
    p: float = 1.0
    cap: int = 2_000_000

    n: int = 0
    n_seen: int = 0  # before `select`
    mean: float = 0.0
    m2: float = 0.0
    min: float = np.inf
    max: float = -np.inf
    # Same moments for the RAW (pre-transform) values. Exact, so the "before" plot and the
    # report quote the true range rather than the subsample's.
    raw_mean: float = 0.0
    raw_m2: float = 0.0
    raw_min: float = np.inf
    raw_max: float = -np.inf
    # Raw range of the values a selection threw away. The dataset applies the transform to
    # ALL values (e.g. dataset_maskformers.py:464 scales unfiltered time), so it matters
    # where the excluded population ends up.
    excl_min: float = np.inf
    excl_max: float = -np.inf
    _buf_t: list[np.ndarray] = field(default_factory=list)
    _buf_raw: list[np.ndarray] = field(default_factory=list)
    _n_buf: int = 0

    def update(self, transformed: np.ndarray, raw: np.ndarray, n_seen: int, excluded: np.ndarray | None = None) -> None:
        self.n_seen += n_seen
        if excluded is not None and excluded.size:
            self.excl_min = min(self.excl_min, float(excluded.min()))
            self.excl_max = max(self.excl_max, float(excluded.max()))
        n_b = transformed.size
        if n_b == 0:
            return

        transformed = transformed.astype(np.float64, copy=False)
        self.mean, self.m2 = _combine(self.n, self.mean, self.m2, transformed)
        self.raw_mean, self.raw_m2 = _combine(self.n, self.raw_mean, self.raw_m2, raw.astype(np.float64, copy=False))
        self.n += n_b

        self.min = min(self.min, float(transformed.min()))
        self.max = max(self.max, float(transformed.max()))
        self.raw_min = min(self.raw_min, float(raw.min()))
        self.raw_max = max(self.raw_max, float(raw.max()))

        self._sample(transformed, raw)

    def _sample(self, transformed: np.ndarray, raw: np.ndarray) -> None:
        # Feed the chunk in cap-sized pieces, thinning between them: appending a whole
        # batch (~8M hits) before thinning would spike memory.
        #
        # The p-filter is applied per piece with the CURRENT p, which is what keeps the
        # sample uniform - p already encodes every thin that happened before this piece
        # arrived, so a late value gets the same overall survival probability as an early
        # one that has since been thinned k times.
        step = self.cap if self.cap else transformed.size
        for start in range(0, transformed.size, step):
            piece = slice(start, start + step)
            t_piece, r_piece = transformed[piece], raw[piece]
            if self.p < 1.0:
                keep = self.rng.random(t_piece.size) < self.p
                t_piece, r_piece = t_piece[keep], r_piece[keep]
            if t_piece.size:
                self._buf_t.append(t_piece.astype(np.float32))
                self._buf_raw.append(r_piece.astype(np.float32))
                self._n_buf += t_piece.size
            while self.cap and self._n_buf > self.cap:
                self._thin()

    def _thin(self) -> None:
        """Halve the buffer and the sampling probability, keeping the sample uniform.

        Halving `p` at the same moment as the buffer is what preserves uniformity: every
        value ever seen has been through the same number of coin flips.
        """
        buf_t = np.concatenate(self._buf_t)
        buf_raw = np.concatenate(self._buf_raw)
        keep = self.rng.random(buf_t.size) < 0.5
        self._buf_t = [buf_t[keep]]
        self._buf_raw = [buf_raw[keep]]
        self._n_buf = int(keep.sum())
        self.p /= 2.0

    @property
    def std(self) -> float:
        if self.n == 0:
            return float("nan")
        return float(np.sqrt(self.m2 / self.n))

    @property
    def raw_std(self) -> float:
        if self.n == 0:
            return float("nan")
        return float(np.sqrt(self.raw_m2 / self.n))

    @property
    def sample(self) -> np.ndarray:
        return np.concatenate(self._buf_t) if self._buf_t else np.empty(0, dtype=np.float32)

    @property
    def sample_raw(self) -> np.ndarray:
        return np.concatenate(self._buf_raw) if self._buf_raw else np.empty(0, dtype=np.float32)

    def quantiles(self, levels) -> dict[float, float]:
        s = self.sample
        if s.size == 0:
            return {q: float("nan") for q in levels}
        vals = np.percentile(s.astype(np.float64), list(levels))
        return dict(zip(levels, (float(v) for v in vals), strict=True))

    def nsigma_hi(self) -> float:
        return abs(self.max - self.mean) / self.std if self.std > 0 else float("inf")

    def nsigma_lo(self) -> float:
        return abs(self.mean - self.min) / self.std if self.std > 0 else float("inf")


# ---------------------------------------------------------------------------------------
# Derivation and transformation
# ---------------------------------------------------------------------------------------
def derive_topocluster_arrays(topos: ak.Array, collection: str) -> dict[str, np.ndarray]:
    """Flatten one batch of topoclusters and derive the geometric quantities.

    Mirrors the "Derive kinematics" block of HGCDataset.__init__ exactly: eta/phi are
    taken from the file when the collection provides them (LayerCluster) and derived from
    x/y/z otherwise (RecHitHGC). If the two drift apart, the statistics stop describing
    what the model is fed at training time.
    """
    fields = TOPO_FIELD_MAP[collection]
    flat = {name: ak.to_numpy(ak.flatten(topos[branch])).astype(np.float64) for name, branch in fields.items()}
    x, y, z = flat["x"], flat["y"], flat["z"]

    rho = np.sqrt(x**2 + y**2)  # distance in the transverse plane
    if "eta" in flat:
        eta = flat["eta"]
    else:
        theta = np.arctan2(rho, z)
        eta = -np.log(np.tan(theta / 2))
    phi = flat["phi"] if "phi" in flat else np.arctan2(y, x)  # phi=0 points vertically upward

    out = {
        "x": x,
        "y": y,
        "z": z,
        "e": flat["energy"],
        "rho": rho,
        "r": np.sqrt(x**2 + y**2 + z**2),
        "eta": eta,
        "sinphi": np.sin(phi),
        "cosphi": np.cos(phi),
    }
    if "time" in flat:
        out["t"] = flat["time"]
    if "em_energy" in flat:
        out["e_em"] = flat["em_energy"]
    return out


def apply_transform(name: str, fn: str | None, raw: np.ndarray) -> np.ndarray:
    """Apply `fn`, refusing to produce nan/-inf silently on out-of-domain input."""
    func, domain = TRANSFORMS[fn]
    if domain == "nonneg":
        bad = raw < 0
    elif domain == "positive":
        bad = raw <= 0
    else:
        bad = None
    if bad is not None and bad.any():
        raise ValueError(
            f"Variable {name!r}: {int(bad.sum())} of {raw.size} values are out of the domain of "
            f"fn={fn!r} (needs {'x >= 0' if domain == 'nonneg' else 'x > 0'}); observed min={raw.min():.6g}. "
            "Use log1p instead of log for values that can reach 0, or add a selection."
        )
    return func(raw)


# ---------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------
def iter_event_batches(files: list[str], batch_events: int, max_events: int | None, topo_collection: str, particle_collection: str | None):
    """Yield (topoclusters, particles_or_None) per batch of events.

    Each parquet file here is a single row group of 1000 events (~42M rechits), so reading
    in event-sized batches is what keeps peak memory bounded. Topoclusters and particles
    are read together so the whole script needs only ONE pass over the files.
    """
    n_events = 0
    for path in files:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        if topo_collection not in names:
            raise ValueError(f"{Path(path).name} has no {topo_collection!r} collection (has: {names})")
        cols = [topo_collection]
        has_particles = particle_collection is not None and particle_collection in names
        if particle_collection is not None and not has_particles:
            print(f"  note: no {particle_collection} collection in {Path(path).name}, skipping the particle fit for it")
        if has_particles:
            cols.append(particle_collection)
        print(f"  reading {Path(path).name} ({pf.metadata.num_rows} events)...", flush=True)

        for batch in pf.iter_batches(batch_size=batch_events, columns=cols):
            take = batch.num_rows
            if max_events is not None:
                take = min(take, max_events - n_events)
                if take <= 0:
                    return
            topos = ak.from_arrow(batch.column(topo_collection))[:take]
            particles = ak.from_arrow(batch.column(particle_collection))[:take] if has_particles else None
            n_events += take
            yield topos, particles
            if max_events is not None and n_events >= max_events:
                return


def derive_particle_arrays(particles: ak.Array, collection: str) -> dict[str, np.ndarray]:
    """Particle-level quantities, matching the particle_data dict in HGCDataset.load_event
    (R derived from the particle position exactly as HGCDataset.__init__ does)."""
    fields = PARTICLE_FIELD_MAP[collection]

    def flat(name):
        return ak.to_numpy(ak.flatten(particles[fields[name]])).astype(np.float64)

    px, py, pz = flat("x"), flat("y"), flat("z")
    phi = flat("phi")
    out = {
        "e": flat("e"),
        "eta": flat("eta"),
        "r": np.sqrt(px**2 + py**2 + pz**2),
        "sinphi": np.sin(phi),
        "cosphi": np.cos(phi),
    }
    if all(f in fields for f in BOUNDARY_FIELDS):
        # entrance-position targets, as HGCDataset builds them: eta/phi derived from the
        # position (the dumped boundaryPhi is broken, see dataset.collections)
        bx, by, bz = flat("boundary_x"), flat("boundary_y"), flat("boundary_z")
        beta = np.arcsinh(bz / np.sqrt(bx**2 + by**2))
        bphi = np.arctan2(by, bx)
        out.update({
            "boundary_x": bx,
            "boundary_y": by,
            "boundary_z": bz,
            "boundary_eta": beta,
            "boundary_sinphi": np.sin(bphi),
            "boundary_cosphi": np.cos(bphi),
        })
    return out


def _accumulate_specs(stats: dict[str, RunningStats], specs: tuple[VarSpec, ...], arrays: dict[str, np.ndarray]) -> None:
    for spec in specs:
        if spec.alias_of is not None:
            continue  # entry is copied at fit time; nothing to measure
        raw_all = arrays[spec.source]
        excluded = None
        if spec.select is not None:
            keep = spec.select(raw_all)
            raw, excluded = raw_all[keep], raw_all[~keep]
        else:
            raw = raw_all
        transformed = apply_transform(spec.name, spec.fn, raw)
        stats[spec.name].update(transformed, raw, n_seen=raw_all.size, excluded=excluded)


def accumulate(files, args, topo_specs, particle_specs) -> tuple[dict[str, RunningStats], dict[str, RunningStats]]:
    """Single pass: accumulate topocluster and particle statistics."""
    rng = np.random.default_rng(args.seed)
    stats = {spec.name: RunningStats(rng=rng, p=args.sample_frac, cap=args.sample_cap) for spec in topo_specs}
    pstats = {spec.name: RunningStats(rng=rng, p=args.sample_frac, cap=args.sample_cap) for spec in particle_specs}
    particle_collection = None if args.no_particles else args.particle_collection

    for topos, particles in iter_event_batches(files, args.batch_events, args.max_events, args.topocluster_collection, particle_collection):
        arrays = derive_topocluster_arrays(topos, args.topocluster_collection)
        _accumulate_specs(stats, topo_specs, arrays)
        if particles is not None:
            _accumulate_specs(pstats, particle_specs, derive_particle_arrays(particles, particle_collection))
        del arrays, topos, particles

    return stats, pstats


# ---------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------
def resolve_bounds_mode(spec: VarSpec, override: str) -> str:
    return spec.bounds if override == "spec" else override


def fit_entry(spec: VarSpec, s: RunningStats, args, fitted: dict[str, dict] | None = None) -> dict:
    """Build the YAML entry for one variable."""
    if spec.alias_of is not None:
        return dict((fitted or {})[spec.alias_of])
    if spec.emit == "identity":
        return {"type": "std", "mean": 0.0, "std": 1.0}

    mode = resolve_bounds_mode(spec, args.bounds)
    if mode == "quantile":
        qs = s.quantiles((args.clip_quantile, 100.0 - args.clip_quantile))
        lo, hi = qs[args.clip_quantile], qs[100.0 - args.clip_quantile]
    else:
        lo, hi = s.min, s.max

    entry = {"type": spec.type}
    if spec.fn:
        entry["fn"] = spec.fn
    entry["mean"] = round(s.mean, 3)
    entry["std"] = round(s.std, 3)
    entry["min"] = round(lo, 3)
    entry["max"] = round(hi, 3)
    return entry


def entry_shift_scale(entry: dict) -> tuple[float, float]:
    """Reproduce VarTransform's shift/scale from an entry, for plotting and validation.

    Uses the *rounded* values that are actually written, so a rounding pathology (e.g.
    min == max after rounding) shows up here rather than at training time.
    """
    if entry["type"] == "std":
        return entry.get("mean", 0.0), entry.get("std", 1.0)
    if entry["type"] == "min_max":
        return entry["min"], entry["max"] - entry["min"]
    return (entry["max"] + entry["min"]) / 2, (entry["max"] - entry["min"]) / 2


def check_outliers(stats: dict[str, RunningStats], args, specs: tuple[VarSpec, ...]) -> tuple[list[str], list[str]]:
    """Return (warnings, failures) for the n_sigma checks on the transformed values."""
    warns, fails = [], []
    for spec in specs:
        if spec.alias_of is not None:
            continue  # nothing measured; the source variable is checked instead
        if spec.name in args.identity_vars:
            continue  # emitted as the identity on request (--identity), so there is no fit to check
        s = stats[spec.name]
        if s.n == 0:
            fails.append(f"{spec.name}: no values accumulated (empty selection or empty input)")
            continue
        warn_thr = args.warn_nsigma
        fail_thr = NSIGMA_FAIL.get(spec.name, NSIGMA_FAIL_DEFAULT)
        for side, nsig, extreme in (("max", s.nsigma_hi(), s.max), ("min", s.nsigma_lo(), s.min)):
            if nsig > fail_thr:
                fails.append(
                    f"{spec.name}: {side}={extreme:.4g} is {nsig:.1f} sigma from mean={s.mean:.4g} "
                    f"(limit {fail_thr:.1f}); transform fn={spec.fn}"
                )
            elif nsig > warn_thr:
                warns.append(f"{spec.name}: {side}={extreme:.4g} is {nsig:.1f} sigma from mean={s.mean:.4g} (warn above {warn_thr:.1f})")
    return warns, fails


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------
def print_report(stats: dict[str, RunningStats], entries: dict[str, dict], args, specs: tuple[VarSpec, ...], label: str) -> None:
    print(f"\n=== {label}: transformed-variable statistics ===")
    header = f"{'var':>8} {'fn':>6} {'n':>13} {'mean':>11} {'std':>10} {'min':>11} {'max':>11}"
    header += "".join(f"{f'q{q:g}':>11}" for q in REPORT_QUANTILES)
    header += f"{'nsig_hi':>9}{'nsig_lo':>9}"
    print(header)
    print("-" * len(header))
    for spec in specs:
        if spec.alias_of is not None:
            print(f"{spec.name:>8} {'(alias of ' + spec.alias_of + ')':>40}")
            continue
        s = stats[spec.name]
        qs = s.quantiles(REPORT_QUANTILES)
        row = f"{spec.name:>8} {str(spec.fn or '-'):>6} {s.n:>13,} {s.mean:>11.4g} {s.std:>10.4g} {s.min:>11.4g} {s.max:>11.4g}"
        row += "".join(f"{qs[q]:>11.4g}" for q in REPORT_QUANTILES)
        row += f"{s.nsigma_hi():>9.1f}{s.nsigma_lo():>9.1f}"
        print(row)

    print(f"\n=== {label}: raw (pre-transform) statistics, exact over all values ===")
    print(f"{'var':>8} {'mean':>11} {'std':>10} {'min':>11} {'max':>11}  {'nsig_hi':>8}")
    for spec in specs:
        if spec.alias_of is not None:
            continue
        s_ = stats[spec.name]
        nsig = abs(s_.raw_max - s_.raw_mean) / s_.raw_std if s_.raw_std > 0 else float("inf")
        print(f"{spec.name:>8} {s_.raw_mean:>11.4g} {s_.raw_std:>10.4g} {s_.raw_min:>11.4g} {s_.raw_max:>11.4g}  {nsig:>8.1f}")

    print(f"\n=== {label}: emitted parameters (what the model will use) ===")
    print(f"{'var':>8} {'type':>12} {'fn':>6} {'bounds':>9} {'shift':>11} {'scale':>11}  note")
    for spec in specs:
        e = entries[spec.name]
        shift, scale = entry_shift_scale(e)
        mode = "identity" if spec.emit == "identity" else resolve_bounds_mode(spec, args.bounds)
        note = spec.comment
        if spec.alias_of is None and spec.select_desc:
            s = stats[spec.name]
            frac = 100.0 * s.n / s.n_seen if s.n_seen else 0.0
            note = f"{note}; select {spec.select_desc} keeps {frac:.2f}%"
        print(f"{spec.name:>8} {e['type']:>12} {str(e.get('fn') or '-'):>6} {mode:>9} {shift:>11.4g} {scale:>11.4g}  {note}")

    # A selection makes the parameters describe the selected population only, but the
    # dataset transforms every value. Say where the excluded values actually land.
    for spec in specs:
        s = stats[spec.name]
        if spec.select is None or not np.isfinite(s.excl_min):
            continue
        try:
            excl = apply_transform(spec.name, spec.fn, np.array([s.excl_min, s.excl_max]))
        except ValueError:
            print(f"\nNOTE: excluded values of {spec.name!r} are outside the domain of fn={spec.fn!r}.")
            continue
        shift, scale = entry_shift_scale(entries[spec.name])
        scaled = (excl - shift) / scale
        n_excl = s.n_seen - s.n
        flag = "WARNING" if max(abs(scaled)) > 2.0 else "NOTE"
        print(
            f"\n{flag}: {spec.name!r} is fitted on '{spec.select_desc}', but the dataset transforms ALL values "
            f"(dataset_maskformers.py:464). The {n_excl:,} excluded values (raw {s.excl_min:.4g}..{s.excl_max:.4g}, "
            f"{100.0 * n_excl / s.n_seen:.2f}% of hits) map to {scaled.min():.4g}..{scaled.max():.4g} after scaling. "
            "Clamping the sentinel, or adding an explicit validity flag feature, would keep the inputs bounded."
        )


def print_transform_alternatives(stats: dict[str, RunningStats], name: str = "e") -> None:
    """Show fitted parameters for every candidate transform of one variable.

    Reuses the raw half of the subsample already drawn in the single data pass, so this
    costs nothing, and it makes switching transform a one-line YAML edit instead of a
    rerun.
    """
    raw = stats[name].sample_raw.astype(np.float64)
    if raw.size == 0:
        return

    print(f"\n=== Alternative transforms for {name!r} (from the subsample) ===")
    print(f"{'fn':>6} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'q0.1':>10} {'q99.9':>10} {'nsig_hi':>9}")
    for fn in (None, "sqrt", "log", "log1p"):
        try:
            t = apply_transform(name, fn, raw)
        except ValueError as exc:
            print(f"{str(fn or '-'):>6}  unusable: {exc}")
            continue
        mean, std = float(t.mean()), float(t.std())
        q_lo, q_hi = (float(v) for v in np.percentile(t, [0.1, 99.9]))
        nsig = abs(t.max() - mean) / std if std > 0 else float("inf")
        print(f"{str(fn or '-'):>6} {mean:>10.4g} {std:>10.4g} {t.min():>10.4g} {t.max():>10.4g} {q_lo:>10.4g} {q_hi:>10.4g} {nsig:>9.1f}")


def report_shared_key_spreads(topo_stats: dict[str, RunningStats], particle_stats: dict[str, RunningStats], args) -> None:
    """Quantify why the two dicts must stay separate: same keys, very different spreads.

    HGCDataset used to push particle e/eta/r through the topocluster entries (they share
    the key names); the ratio printed here is the size of that mismatch in RAW units,
    using the spread rather than the mean because eta is symmetric about zero.
    """
    print(f"\n=== Shared keys: {args.topocluster_collection} inputs vs {args.particle_collection} targets (raw spreads) ===")
    for key in ("e", "eta", "r"):
        ts, ps = topo_stats.get(key), particle_stats.get(key)
        if ts is None or ps is None or ts.n == 0 or ps.n == 0:
            continue
        t_scale, p_scale = abs(ts.raw_std), abs(ps.raw_std)
        ratio = max(p_scale / t_scale, t_scale / p_scale) if t_scale > 0 and p_scale > 0 else float("inf")
        print(f"{key:>8}: topocluster std {t_scale:.4g}  particle std {p_scale:.4g}  ratio {ratio:.1f}x -> separate entries in {Path(args.output).name} / {Path(args.particle_output).name}")


# ---------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------
def plot_variable(spec: VarSpec, s: RunningStats, entry: dict, out_dir: Path, args, prefix: str = "", unit: str = "rechits") -> None:
    """Write two PNGs per variable: the raw distribution and the scaled one.

    Both come from the bounded subsample, so plotting costs no extra pass and no extra
    memory. The 'after' panel is the most direct check that the parameters are sane: with
    the old (raw-space) parameters, e_after is a single spike at -0.998.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = s.sample_raw.astype(np.float64)
    if raw.size == 0:
        return
    transformed = apply_transform(spec.name, spec.fn, raw)
    shift, scale = entry_shift_scale(entry)
    scaled = (transformed - shift) / scale

    qs = s.quantiles(REPORT_QUANTILES)
    sel = f"\nselection: {spec.select_desc} ({100.0 * s.n / s.n_seen:.2f}% of {unit})" if spec.select_desc and s.n_seen else ""

    panels = (
        (
            "before",
            raw,
            "raw value",
            f"{prefix}{spec.name} - before scaling",
            # Exact accumulated moments, not the subsample's - the true extremes are the
            # whole point of the outlier check.
            f"fn: none (raw)\nn = {s.n:,}\nmean = {s.raw_mean:.4g}\nstd = {s.raw_std:.4g}\nmin = {s.raw_min:.4g}\nmax = {s.raw_max:.4g}{sel}",
        ),
        (
            "after",
            scaled,
            "scaled value",
            f"{prefix}{spec.name} - after scaling",
            (
                f"fn: {spec.fn or 'none'}\ntype: {entry['type']}\n"
                f"shift = {shift:.4g}\nscale = {scale:.4g}\n"
                f"n = {s.n:,}\n"
                f"transformed mean = {s.mean:.4g}\ntransformed std = {s.std:.4g}\n"
                f"q0.1 = {qs[0.1]:.4g}\nq50 = {qs[50.0]:.4g}\nq99.9 = {qs[99.9]:.4g}\n"
                f"nsigma(max) = {s.nsigma_hi():.1f}{sel}"
            ),
        ),
    )

    for tag, values, xlabel, title, textbox in panels:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        v_lo, v_hi = float(values.min()), float(values.max())
        # A strictly-positive variable spanning decades (the raw rechit energy) is
        # unreadable on a linear axis. Log-spaced bins show the whole range, so there is
        # no need to clip the display at all.
        if v_lo > 0 and v_hi / v_lo > 100:
            bins = np.logspace(np.log10(v_lo), np.log10(v_hi), 101)
            ax.set_xscale("log")
            clipped = False
        else:
            lo, hi = (float(v) for v in np.percentile(values, [args.clip_quantile, 100.0 - args.clip_quantile]))
            clipped = hi > lo and (v_lo < lo or v_hi > hi)
            if hi <= lo:  # degenerate: a (near-)constant variable
                lo, hi = v_lo, v_hi
                if hi <= lo:
                    lo, hi = lo - 0.5, hi + 0.5
            bins = np.linspace(lo, hi, 101)
        ax.hist(values, bins=bins, color="#3b76af", log=True)
        if tag == "after":
            for edge in (-1.0, 1.0):
                ax.axvline(edge, ls="--", lw=1.0, color="crimson")
            ax.axvline(0.0, ls=":", lw=1.0, color="grey")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{unit} (log scale)")
        ax.set_title(title)
        ax.text(
            0.98,
            0.97,
            textbox,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "grey"},
        )
        if clipped:
            ax.text(
                0.02,
                0.97,
                f"display range clipped to [q{args.clip_quantile:g}, q{100 - args.clip_quantile:g}]\ntrue range [{v_lo:.4g}, {v_hi:.4g}]",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                family="monospace",
                color="dimgrey",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "lightgrey"},
            )
        fig.tight_layout()
        path = out_dir / f"{prefix}{spec.name}_{tag}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
    print(f"  wrote {out_dir / f'{prefix}{spec.name}_before.png'} and {out_dir / f'{prefix}{spec.name}_after.png'}")


# ---------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------
def write_yaml(entries: dict[str, dict], stats: dict[str, RunningStats], files: list[str], args, out_path: Path | None, collection: str, role: str) -> str:
    n_items = max((s.n_seen for s in stats.values()), default=0)
    header = [
        f"# Generated by {Path(__file__).name} on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"# collection: {collection} ({role})",
        f"# input: {args.input_dir} ({len(files)} files, {n_items:,} {collection} entries" + (f", capped at {args.max_events} events" if args.max_events else "") + ")",
        f"# bounds: {args.bounds} (clip quantile q{args.clip_quantile:g}/q{100 - args.clip_quantile:g}), subsample frac {args.sample_frac}, cap {args.sample_cap}, seed {args.seed}",
        "# NOTE: mean/std/min/max are statistics of the TRANSFORMED variable - `fn` is",
        "#       applied first, then (x - shift)/scale. See hepattn.utils.scaling.",
        "",
    ]
    body = yaml.dump(entries, default_flow_style=False, sort_keys=False)
    text = "\n".join(header) + body
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    return text


def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input-dir",
        default="/home/export/mmarcheg/store/ml4reco/data_maskformers/parquet_out_single-files",
        help="Directory containing parquet files, or a single .parquet file",
    )
    p.add_argument("--topocluster-collection", choices=tuple(TOPO_FIELD_MAP), default="RecHitHGC", help="Input node collection the node dict is fitted on")
    p.add_argument("--particle-collection", choices=PARTICLE_COLLECTIONS, default="MergedSimCluster", help="Target collection the particle dict is fitted on")
    p.add_argument("--output", default=None, help="Node scale-dict YAML (default: configs/<DEFAULT_OUTPUT[topocluster-collection]>)")
    p.add_argument("--particle-output", default=None, help="Particle scale-dict YAML (default: configs/<DEFAULT_OUTPUT[particle-collection]>)")
    p.add_argument("--max-files", type=int, default=None, help="Only read the first N files")
    p.add_argument("--max-events", type=int, default=None, help="Only read the first N events in total (cheap smoke tests)")
    p.add_argument("--batch-events", type=int, default=200, help="Events per read batch (bounds peak memory)")
    p.add_argument("--threads", type=int, default=1, help="Thread count for pyarrow/BLAS. Keep at 1 on the gateway (60%% CPU policy)")
    p.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Initial Bernoulli subsample probability for quantiles and plots. The default of 1.0 relies on "
        "--sample-cap thinning, which keeps the sample uniform and bounded; lower it only to save work",
    )
    p.add_argument("--sample-cap", type=int, default=2_000_000, help="Max subsampled values kept per variable (0 = uncapped)")
    p.add_argument("--clip-quantile", type=float, default=0.1, help="Quantile (in %%) used for min, with 100-q used for max")
    p.add_argument("--bounds", choices=("spec", "quantile", "extremes"), default="spec", help="Override how min/max are chosen for every variable")
    p.add_argument("--warn-nsigma", type=float, default=NSIGMA_WARN_DEFAULT, help="Warn when an extreme is this many sigma from the mean")
    p.add_argument("--allow-outliers", action="store_true", help="Downgrade n_sigma failures to warnings")
    p.add_argument("--seed", type=int, default=0, help="Seed for the subsampling RNG")
    p.add_argument("--plot-dir", default=str(repo_root / "plots" / "norm_par"), help="Where to write the before/after plots")
    p.add_argument("--no-plots", action="store_true", help="Skip plotting")
    p.add_argument("--no-particles", action="store_true", help="Skip the particle dict (only fit and write the node dict)")
    p.add_argument(
        "--identity",
        nargs="+",
        default=[],
        metavar="VAR",
        help="Write these variables of either dict as the identity (type std, mean 0, std 1) instead of fitting them, "
        "for variables the files cannot be fitted on: e.g. boundary_z on the toy calorimeter, where every entry "
        "point lies on the front face (a zero scale otherwise). Aliases (t_filt) follow their source",
    )
    p.add_argument(
        "--identity-time",
        action="store_true",
        help="Same as --identity t: for files whose time is the -99 sentinel for every topocluster (the toy "
        "calorimeter); time must then not be a model input",
    )
    p.add_argument("--no-alternatives", action="store_true", help="Skip the alternative-transform table for 'e' (saves one extra pass)")
    p.add_argument("--dry-run", action="store_true", help="Print the YAML instead of writing it")
    args = p.parse_args(argv)
    if args.output is None:
        args.output = str(repo_root / "configs" / DEFAULT_OUTPUT[args.topocluster_collection])
    if args.particle_output is None:
        args.particle_output = str(repo_root / "configs" / DEFAULT_OUTPUT[args.particle_collection])
    if not args.no_particles and Path(args.output).resolve() == Path(args.particle_output).resolve():
        p.error("--output and --particle-output must be different files: the node and particle dicts share key names")
    args.identity_vars = set(args.identity) | ({"t"} if args.identity_time else set())
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    pa.set_cpu_count(args.threads)
    pa.set_io_thread_count(args.threads)

    input_path = Path(args.input_dir)
    files = [str(input_path)] if input_path.is_file() else sorted(glob(f"{args.input_dir}/*.parquet"))
    if not files:
        print(f"ERROR: no parquet files found in {args.input_dir!r}", file=sys.stderr)
        return 2
    if args.max_files is not None:
        files = files[: args.max_files]
    print(f"Found {len(files)} file(s) in {args.input_dir}")
    print(f"Collections: topocluster={args.topocluster_collection} -> {args.output}")
    if not args.no_particles:
        print(f"             particle={args.particle_collection} -> {args.particle_output}")
    print(f"Threads: {args.threads}; batch: {args.batch_events} events; max_events: {args.max_events}")

    topo_specs = topo_var_specs(args.topocluster_collection)
    particle_specs = () if args.no_particles else particle_var_specs(args.particle_collection)
    # Variables emitted as the identity on request: still measured and reported, but neither fitted nor
    # outlier-checked (an alias such as t_filt copies its source's entry)
    unknown = args.identity_vars - {spec.name for spec in topo_specs + particle_specs if spec.alias_of is None}
    if unknown:
        print(f"ERROR: --identity names variables this collection pair does not have: {sorted(unknown)}", file=sys.stderr)
        return 2
    if args.identity_vars:
        print(f"Emitted as the identity, not fitted: {sorted(args.identity_vars)}")
    topo_specs, particle_specs = (
        tuple(replace(spec, emit="identity", comment="identity (--identity)") if spec.name in args.identity_vars else spec for spec in specs)
        for specs in (topo_specs, particle_specs)
    )
    stats, pstats = accumulate(files, args, topo_specs, particle_specs)

    # (label, specs, stats, output path, plot prefix, unit) for each dict to fit
    # plot files are prefixed with the collection so several (topocluster, particle) pairs can share one --plot-dir
    jobs = [(args.topocluster_collection, topo_specs, stats, Path(args.output), f"{args.topocluster_collection}_", "topoclusters")]
    if particle_specs:
        if all(s.n == 0 for s in pstats.values()):
            print(f"\nERROR: no {args.particle_collection} entries seen; use --no-particles or a file that has the collection", file=sys.stderr)
            return 2
        jobs.append((args.particle_collection, particle_specs, pstats, Path(args.particle_output), f"{args.particle_collection}_", "particles"))

    all_entries: dict[str, dict[str, dict]] = {}
    for label, specs, st, _out, _prefix, _unit in jobs:
        entries: dict[str, dict] = {}
        for spec in specs:  # ordered, so an alias always sees its source
            entries[spec.name] = fit_entry(spec, st[spec.name], args, entries)
        for name, entry in entries.items():
            if entry["type"] in {"min_max", "min_max_sym"} and entry["max"] == entry["min"]:
                print(
                    f"ERROR: {label} variable {name!r} has min == max == {entry['min']} after rounding, which gives a zero scale.",
                    file=sys.stderr,
                )
                return 1
        all_entries[label] = entries
        print_report(st, entries, args, specs, label)

    if not args.no_alternatives:
        print_transform_alternatives(stats)
    if particle_specs:
        report_shared_key_spreads(stats, pstats, args)

    fails_total = []
    for label, specs, st, _out, _prefix, _unit in jobs:
        warns, fails = check_outliers(st, args, specs)
        if warns:
            print(f"\n=== {label}: outlier warnings ===")
            for w in warns:
                print(f"WARN  {w}")
        if fails:
            print(f"\n=== {label}: outlier failures ===")
            for f in fails:
                print(f"FAIL  {f}")
        fails_total += fails

    if not args.no_plots:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Plots -> {plot_dir} ===")
        for label, specs, st, _out, prefix, unit in jobs:
            for spec in specs:
                if spec.alias_of is not None:
                    continue  # identical to its source variable's plots
                plot_variable(spec, st[spec.name], all_entries[label][spec.name], plot_dir, args, prefix=prefix, unit=unit)

    # ru_maxrss is kilobytes on Linux but bytes on macOS
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    print(f"\nElapsed {time.time() - t_start:.1f} s, peak RSS {peak_mb:.0f} MB")

    if fails_total and not args.allow_outliers:
        print(
            f"\nRefusing to write: {len(fails_total)} variable(s) exceeded their n_sigma limit.\n"
            "Fix the variable (transform, selection) or raise its NSIGMA_FAIL entry if the tail is physical, "
            "or pass --allow-outliers.",
            file=sys.stderr,
        )
        return 1

    for label, _specs, st, out, _prefix, _unit in jobs:
        role = "node inputs" if out == Path(args.output) else "regression targets"
        text = write_yaml(all_entries[label], st, files, args, None if args.dry_run else out, label, role)
        if args.dry_run:
            print(f"\n=== {label} YAML -> {out} (dry run, not written) ===")
            print(text)
        else:
            print(f"\nWrote {label} scale dict to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
