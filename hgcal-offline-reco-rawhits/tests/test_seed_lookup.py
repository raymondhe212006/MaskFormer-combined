"""Tests for the seed lookup of ticlSimTracksters: SimCluster-seeded particles (every tau decay
product) are looked up directly, CaloParticle-seeded ones (an unconverted photon) through the
SimClusters of their CaloParticle (dataset.caloparticle_seeds).

    PYTHONPATH=src python -m pytest tests/test_seed_lookup.py -q

The HGCDataset part needs torch (the project apptainer image); the helper part only awkward.
"""

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataset.caloparticle_seeds import resolve_caloparticle_seeds  # noqa: E402
from dataset.collections import PARTICLE_SEED_LOOKUP  # noqa: E402

P, S = "ticlSimTracksters", "SimCluster"


def _events():
    """Four events, per-particle (seedIndex, seedIsCaloParticle) and per-SimCluster
    (CaloPartIdx, boundaryEnergy, pdgId, eventId):

    0: tau-like, every particle seeded by a SimCluster;
    1: an unconverted photon seeded by CaloParticle 1, whose SimClusters are rows 0 (e-, 0.3 GeV)
       and 2 (gamma, 9.7 GeV), plus a SimCluster-seeded pion;
    2: a CaloParticle-seeded particle whose CaloParticle row (5) exceeds the SimCluster count;
    3: no particle.
    """
    seed_idx = ak.Array([[2, 0, 1], [1, 1], [5], []])
    is_cp = ak.Array([[False, False, False], [True, False], [True], []])
    group = ak.Array([[0, 0, 0], [1, 0, 1], [5, 5], []])
    energy = ak.Array([[1.0, 2.0, 3.0], [0.3, 4.0, 9.7], [2.0, 8.0], []])
    pdg = ak.Array([[211, 22, 11], [11, 211, 22], [-11, 22], []])
    event_id = ak.Array([[0, 0, 0], [3, 0, 3], [0, 0], []])
    return seed_idx, is_cp, group, energy, pdg, event_id


def test_resolve_most_energetic_simcluster():
    seed_idx, is_cp, group, energy, _, _ = _events()
    resolved, n_unresolved = resolve_caloparticle_seeds(seed_idx, is_cp, group, energy)
    assert n_unresolved == 0
    assert ak.to_list(resolved) == [[2, 0, 1], [2, 1], [1], []]
    assert str(ak.type(resolved)).endswith("int64")


def test_resolve_first_simcluster_without_energy():
    seed_idx, is_cp, group, _, _, _ = _events()
    resolved, n_unresolved = resolve_caloparticle_seeds(seed_idx, is_cp, group)
    assert n_unresolved == 0
    assert ak.to_list(resolved) == [[2, 0, 1], [0, 1], [0], []]


def test_resolve_is_identity_without_caloparticle_seeds():
    seed_idx, _, group, energy, _, _ = _events()
    no_cp = ak.zeros_like(seed_idx, dtype=bool)
    resolved, n_unresolved = resolve_caloparticle_seeds(seed_idx, no_cp, group, energy)
    assert n_unresolved == 0
    assert ak.to_list(resolved) == ak.to_list(seed_idx)


def test_resolve_reports_caloparticles_without_simclusters():
    seed_idx, is_cp, group, energy, _, _ = _events()
    group = ak.Array([[0, 0, 0], [1, 0, 1], [4, 4], []])   # event 2: CaloParticle 5 has no SimCluster
    resolved, n_unresolved = resolve_caloparticle_seeds(seed_idx, is_cp, group, energy)
    assert n_unresolved == 1
    assert ak.to_list(resolved) == [[2, 0, 1], [2, 1], [-1], []]


def _frame(with_group=True, with_energy=True):
    seed_idx, is_cp, group, energy, pdg, event_id = _events()
    particle = {f"{P}_seedIndex": seed_idx, f"{P}_seedIsCaloParticle": is_cp}
    seeds = {
        f"{S}_pdgId": pdg,
        f"{S}_eventId": event_id,
        f"{S}_bunchCrossing": ak.zeros_like(pdg),
    }
    if with_group:
        seeds[f"{S}_CaloPartIdx"] = group
    if with_energy:
        seeds[f"{S}_boundaryEnergy"] = energy
    record = lambda d: ak.zip(d, depth_limit=1)  # noqa: E731
    return ak.zip({P: record(particle), S: record(seeds)}, depth_limit=1)


def _dataset(df):
    HGCDataset = pytest.importorskip("dataset.dataset_maskformers").HGCDataset
    ds = HGCDataset.__new__(HGCDataset)
    ds.df = df
    ds.particle = df[P]
    ds.particle_collection = P
    ds.filepath = "synthetic"
    return ds


def test_dataset_lookup_resolves_caloparticle_seeds():
    ds = _dataset(_frame())
    ds._lookup_seed_fields(PARTICLE_SEED_LOOKUP[P])
    assert ak.to_list(ds.particle["particle_pdgId"]) == [[11, 211, 22], [22, 211], [22], []]
    assert ak.to_list(ds.particle["particle_eventId"]) == [[0, 0, 0], [3, 0], [0], []]
    assert ak.to_list(ds.particle["particle_bunchCrossing"]) == [[0, 0, 0], [0, 0], [0], []]


def test_dataset_lookup_unchanged_for_simcluster_seeded_only():
    df = _frame()
    df[P, f"{P}_seedIsCaloParticle"] = ak.zeros_like(df[P][f"{P}_seedIndex"], dtype=bool)
    ds = _dataset(df)
    # event 2 now claims a SimCluster row 5 that does not exist: the range check still fires
    with pytest.raises(ValueError, match="out of range"):
        ds._lookup_seed_fields(PARTICLE_SEED_LOOKUP[P])
    df = df[:2]
    ds = _dataset(df)
    ds._lookup_seed_fields(PARTICLE_SEED_LOOKUP[P])
    assert ak.to_list(ds.particle["particle_pdgId"]) == [[11, 211, 22], [211, 211]]


def test_dataset_lookup_needs_caloparticle_row_branch():
    ds = _dataset(_frame(with_group=False))
    with pytest.raises(ValueError, match="seeded by a CaloParticle"):
        ds._lookup_seed_fields(PARTICLE_SEED_LOOKUP[P])
