#!/usr/bin/env python3
"""Convert hg_toy_calorimeter event sets to the NanoML parquet schema that HGCDataset reads.

Input: one ra_pickles directory written by the toy event builder (ml4reco/toy_replication/scripts/build_events.py).
Each pickle is one event, the EventGenerator result dict with every per-hit array, plus result['particles'], the merged
particle table whose row k is the particle with truth_assignment == k. Units there are the EventGenerator's: hit
positions in cm, particle positions in mm, energies in GeV, hit energies already calibrated by the builder.

Output: parquet files of --events-per-file events, one row per event. RecHitHGC and SimCluster are structs of list
columns under their NanoML branch names (dataset.collections), so the pair (RecHitHGC, SimCluster) loads without a
new collection:

  RecHitHGC_x/_y/_z (cm), _energy (calibrated GeV), _time = -99 (the toy has no time: the "no valid time" sentinel of
  the CMS files), _detId = sensor index, _SimClusterBestMatchIdx = row of the SimCluster collection (-1 noise; the
  particle with the largest deposit in the sensor), _SimClusterBestMatchQual = 1 for assigned hits and 0 for noise
  (the builder keeps no per-hit energy share), and two extra branches the model ignores: _layer (active layer
  number) and _area (cm^2).
  SimCluster_impactPoint_x/_y/_z (cm) = where the particle crosses the calorimeter front face (z = CALO_FRONT_Z),
  i.e. the HGCAL boundary crossing of NanoML: the straight line (no magnetic field) from the particle's vertex
  along its momentum direction. _impactPoint_eta/_phi are derived from that point. The point the simulation
  records (particles_first_active_impact_position_*) is kept as the extra _firstImpact_x/_y/_z (cm): it is the
  first energy deposit in ANY volume, so for a charged particle it is the end of its first 1.6 m step through the
  Geant4 world (~150 cm in front of the calorimeter), not a point on the calorimeter. A particle whose vertex is
  at or behind the face, or that does not move towards it, keeps the recorded point (counted in the summary).
  _recEnergy = _boundaryEnergy = the particle's true energy (particles_kinetic_energy, the sum over the
  constituents of a merged particle: the target of the paper and of create_dataset.py, whose t_energy is
  truth_assignment_energy), _pdgId, _eventId (0 signal, 1 pileup = particles_only_minbias), _bunchCrossing = 0,
  _trackIdAtBoundary = _CaloPartIdx = the particle's row in result['particles'], _isPileup, and the extra _merged
  (several close-by particles merged into one by the event builder). Optional, written when the event builder
  provides the column (OPTIONAL_SIMCLUSTER_BRANCHES): _truthAssistedEnergy, the calibrated energy of the particle's
  rechits (a truth-assisted clustering baseline), and _nHits, their number.
  Top level: event_id (index of the event in the input directory) and set_name.

Only particles with at least one rechit become SimClusters, as in the paper ("truth particles kept only with >= 1
hit"): a particle whose Geant4 deposits all fall below the threshold or lose every sensor to a bigger depositor is
dropped (~0.1 per PU-free and ~3 per pileup event). --keep-hitless-particles keeps every row of the particle table.

Floats are stored as float32 and indices as int32, as in the NanoML files; the source arrays are float64/int64, so a
stored float is the float32 rounding of the source value (<= 2^-24 relative). Each file is written under a
.partial name, re-read and compared with the source events, and renamed only if the check passes (--no-check
skips it):
  - copied branches equal float32/int32 of the source bit for bit;
  - BestMatchIdx maps back to truth_assignment through CaloPartIdx, and the SimCluster rows are exactly the
    particles with >= 1 rechit (every row with --keep-hitless-particles);
  - the derived impact point and its eta/phi equal the float32 of the float64 formulas bit for bit, the impact
    point lies on the front face (unless it is a fallback), and eta/phi agree with a recomputation from the stored
    float32 position within ETA_PHI_TOL;
  - the optional branches equal float32/int32 of the builder's columns, _nHits equals the number of stored hits
    matched to the particle, and _truthAssistedEnergy their summed stored energy within TRUTH_SUM_REL_TOL.
A summary (counts, dropped particles, check results, files) is written to <output-dir>/conversion_summary.json.

Sector events (--sectors N, default off). Each event is split into N azimuthal sectors and written as N rows, with
the top-level branch `sector` next to event_id: the toy's version of the 120-degree sectors of the HGCAL L1 trigger
(N = 3). A hit belongs to sector k if the azimuth of its sensor, phi = atan2(y, x) (a ray through the beam axis),
minus --sector-offset-deg lies in [k, k + 1) x 360/N degrees; a sensor centred exactly on a boundary (5,004 of the
778,712 toy sensors) belongs to the sector that starts there. The boundaries sit at 0 / 120 / 240 degrees of the toy
frame, whose orientation relative to CMS x-y is arbitrary (one z > 0 endcap, square sensors). A sector row holds the
sector's hits only, with BestMatchIdx pointing into that row's SimClusters, and the SimClusters are the particles with
>= 1 hit in the sector. The impact point and every other particle branch are the full particle's. Three extra branches:
_sectorEnergyFraction, the share of the particle's hit energy that lies in this sector, and _sectorNHits, its hits
there, and _fullEnergy, its full true energy. The regression target _recEnergy is the full true energy
(--sector-target full, default) or the in-sector hit energy (--sector-target deposited); _boundaryEnergy stays the
full true energy. With --sector-min-share X a particle is a SimCluster of a sector only if >= X of its hit energy is
there; its hits in a sector below the cut are noise (the sliver hits of a shower spilling over a boundary). The sector check
compares every row with the event's full conversion, hit by hit via the sensor index, and recomputes the sector of
every stored hit from its float32 position. The top-level `sector_has_leading` marks, per event, the one sector that
holds the largest share of the hit energy of the event's most energetic SimCluster (by true energy). On single-particle
evaluation sets it flags the sector of the particle; the other sectors are particle-free (fake-rate) tests.
The sector rule itself (sector_of, split_into_sectors) is in toy_sectors.py next to this file: numpy only, so
readers can import it without ra_pickles.

Needs numpy, pyarrow, awkward and ra_pickles (ml4reco/toy_replication/pydeps). Run real sets through Slurm.

Example:
    python src/scripts/dataset/toy_to_parquet.py --input <toy>/data/events/pufree --output-dir <toy>/data/parquet/pufree
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
from toy_sectors import SECTOR_SIMCLUSTER_BRANCHES, sector_of, split_into_sectors

try:
    import ra_pickles
except ImportError:
    sys.exit("ra_pickles not found: put ml4reco/toy_replication/pydeps (ra-pickles 0.9.1.5) on PYTHONPATH")

TIME_SENTINEL = -99.0
# Front face of the toy calorimeter (CaloV3Generator: calo_start_z = 3.2 m), in mm like the particle table.
CALO_FRONT_Z = 3200.0

# Branch order of the written collections.
RECHIT_BRANCHES = (
    "RecHitHGC_x",
    "RecHitHGC_y",
    "RecHitHGC_z",
    "RecHitHGC_energy",
    "RecHitHGC_time",
    "RecHitHGC_detId",
    "RecHitHGC_SimClusterBestMatchIdx",
    "RecHitHGC_SimClusterBestMatchQual",
    "RecHitHGC_layer",
    "RecHitHGC_area",
)
SIMCLUSTER_BRANCHES = (
    "SimCluster_impactPoint_x",
    "SimCluster_impactPoint_y",
    "SimCluster_impactPoint_z",
    "SimCluster_impactPoint_eta",
    "SimCluster_impactPoint_phi",
    "SimCluster_recEnergy",
    "SimCluster_boundaryEnergy",
    "SimCluster_pdgId",
    "SimCluster_eventId",
    "SimCluster_bunchCrossing",
    "SimCluster_trackIdAtBoundary",
    "SimCluster_CaloPartIdx",
    "SimCluster_isPileup",
    "SimCluster_merged",
    "SimCluster_firstImpact_x",
    "SimCluster_firstImpact_y",
    "SimCluster_firstImpact_z",
)
# Written only when the event builder provides the source column (toy_replication build_events.py since
# 2026-10-20 evening; older builds lack them): branch -> (particle-table column, stored dtype). A set must have them
# in every event or in none.
OPTIONAL_SIMCLUSTER_BRANCHES = {
    "SimCluster_truthAssistedEnergy": ("particles_truth_assisted_energy", np.float32),
    "SimCluster_nHits": ("particles_n_hits", np.int32),
}

# Largest |stored eta/phi - eta/phi recomputed from the stored float32 position| the check accepts. The float32
# position carries a relative error <= 6e-8, which moves eta by ~1e-7 and phi by ~1e-7 rad; 1e-5 leaves room for
# float32 rounding of the angles themselves without hiding a real error (one sensor is >= 0.02 wide in eta/phi).
ETA_PHI_TOL = 1e-5
# Largest distance (cm) of a stored front-face crossing from the face plane and from the particle's line. The float32
# position is good to ~1e-5 cm at |x| < 200 cm; the smallest sensor is ~0.65 cm wide.
FACE_TOL_CM = 1e-4
LINE_TOL_CM = 1e-3
# Largest relative difference between SimCluster_truthAssistedEnergy (float32 of the builder's float64 sum) and the sum
# of the particle's stored float32 hit energies: each hit carries <= 2^-24 relative rounding.
TRUTH_SUM_REL_TOL = 1e-5


def as_int32(values, name: str) -> np.ndarray:
    """Cast to int32, refusing anything that would not survive the cast exactly."""
    values = np.asarray(values)
    out = values.astype(np.int32)
    if not np.array_equal(out, values):
        raise ValueError(f"{name}: values do not fit int32 exactly (range {values.min()}..{values.max()})")
    return out


def impact_eta_phi(x, y, z):
    """eta and phi of a point (the same formulas as dataset.collections.boundary_eta_phi)."""
    return np.arcsinh(z / np.hypot(x, y)), np.arctan2(y, x)


def impact_points(particles: dict, rows: np.ndarray):
    """(front-face crossing, recorded first impact, fallback mask) of the given particle-table rows, in mm.

    The crossing is vertex + t * direction with t = (CALO_FRONT_Z - vertex_z) / direction_z. A particle whose
    vertex is at or behind the face, or whose direction_z <= 0, keeps its recorded first impact (fallback)."""
    vertex = np.stack([np.asarray(particles[f"particles_vertex_position_{c}"], dtype=np.float64)[rows] for c in "xyz"], axis=1)
    direction = np.stack([np.asarray(particles[f"particles_momentum_direction_{c}"], dtype=np.float64)[rows] for c in "xyz"], axis=1)
    recorded = np.stack([np.asarray(particles[f"particles_first_active_impact_position_{c}"], dtype=np.float64)[rows] for c in "xyz"], axis=1)
    reaches = (direction[:, 2] > 0) & (vertex[:, 2] < CALO_FRONT_Z)
    t = (CALO_FRONT_Z - vertex[:, 2]) / np.where(reaches, direction[:, 2], 1.0)
    face = np.where(reaches[:, None], vertex + t[:, None] * direction, recorded)
    return face, recorded, ~reaches


def optional_branches(event: dict) -> tuple[str, ...]:
    """The OPTIONAL_SIMCLUSTER_BRANCHES whose source column this event's particle table has."""
    return tuple(b for b, (column, _) in OPTIONAL_SIMCLUSTER_BRANCHES.items() if column in event["particles"])


def simcluster_rows(event: dict, keep_hitless: bool) -> np.ndarray:
    """Rows of result['particles'] that become SimClusters, in table order."""
    if keep_hitless:
        return np.arange(len(event["particles"]["particles_kinetic_energy"]))
    truth = np.asarray(event["truth_assignment"])
    return np.unique(truth[truth >= 0])


def convert_event(event: dict, keep_hitless: bool, optional: tuple[str, ...] = ()) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """One event as (RecHitHGC branches, SimCluster branches incl. the `optional` ones, particle-table rows without a
    rechit, per SimCluster whether its impact point fell back to the recorded first impact)."""
    truth = np.asarray(event["truth_assignment"])
    particles = event["particles"]
    n_table = len(particles["particles_kinetic_energy"])
    if truth.size and truth.max() >= n_table:
        raise ValueError("truth_assignment points past the particle table")
    rows = simcluster_rows(event, keep_hitless)
    hitless = np.setdiff1d(np.arange(n_table), np.unique(truth[truth >= 0]))

    # table row -> SimCluster row; noise stays -1
    to_simcluster = np.full(n_table, -1, dtype=np.int64)
    to_simcluster[rows] = np.arange(len(rows))
    assigned = truth >= 0
    match_idx = np.where(assigned, to_simcluster[np.where(assigned, truth, 0)], -1)

    layer = np.asarray(event["rechit_layer"])
    if not np.array_equal(layer, np.round(layer)):
        raise ValueError("rechit_layer is not integer-valued")
    n_hits = len(truth)
    rechits = {
        "RecHitHGC_x": np.asarray(event["rechit_x"]).astype(np.float32),
        "RecHitHGC_y": np.asarray(event["rechit_y"]).astype(np.float32),
        "RecHitHGC_z": np.asarray(event["rechit_z"]).astype(np.float32),
        "RecHitHGC_energy": np.asarray(event["rechit_energy"]).astype(np.float32),
        "RecHitHGC_time": np.full(n_hits, TIME_SENTINEL, dtype=np.float32),
        "RecHitHGC_detId": as_int32(event["rechit_sensor_idx"], "rechit_sensor_idx"),
        "RecHitHGC_SimClusterBestMatchIdx": match_idx.astype(np.int32),
        "RecHitHGC_SimClusterBestMatchQual": assigned.astype(np.float32),
        "RecHitHGC_layer": as_int32(layer, "rechit_layer"),
        "RecHitHGC_area": np.asarray(event["rechit_area"]).astype(np.float32),
    }

    # impact point = front-face crossing; mm in the particle table, cm in NanoML (and for the rechits)
    face, recorded, fallback = impact_points(particles, rows)
    x, y, z = (face[:, k] / 10.0 for k in range(3))
    eta, phi = impact_eta_phi(x, y, z)
    if not (np.isfinite(eta).all() and np.isfinite(phi).all()):
        raise ValueError("impact point on the beam axis: eta/phi undefined")
    true_energy = np.asarray(particles["particles_kinetic_energy"])[rows].astype(np.float32)
    only_minbias = np.asarray(particles["particles_only_minbias"], dtype=bool)[rows]
    simclusters = {
        "SimCluster_impactPoint_x": x.astype(np.float32),
        "SimCluster_impactPoint_y": y.astype(np.float32),
        "SimCluster_impactPoint_z": z.astype(np.float32),
        "SimCluster_impactPoint_eta": eta.astype(np.float32),
        "SimCluster_impactPoint_phi": phi.astype(np.float32),
        "SimCluster_recEnergy": true_energy,
        "SimCluster_boundaryEnergy": true_energy.copy(),
        "SimCluster_pdgId": as_int32(np.asarray(particles["particles_pdgid"])[rows], "particles_pdgid"),
        "SimCluster_eventId": only_minbias.astype(np.int32),
        "SimCluster_bunchCrossing": np.zeros(len(rows), dtype=np.int32),
        "SimCluster_trackIdAtBoundary": rows.astype(np.int32),
        "SimCluster_CaloPartIdx": rows.astype(np.int32),
        "SimCluster_isPileup": only_minbias,
        "SimCluster_merged": np.asarray(particles["merged"], dtype=bool)[rows],
        "SimCluster_firstImpact_x": (recorded[:, 0] / 10.0).astype(np.float32),
        "SimCluster_firstImpact_y": (recorded[:, 1] / 10.0).astype(np.float32),
        "SimCluster_firstImpact_z": (recorded[:, 2] / 10.0).astype(np.float32),
    }
    for branch in optional:
        column, dtype = OPTIONAL_SIMCLUSTER_BRANCHES[branch]
        if column not in particles:
            raise ValueError(f"{column} is missing in an event of a set whose first event has it")
        values = np.asarray(particles[column])[rows]
        simclusters[branch] = as_int32(values, column) if dtype == np.int32 else values.astype(dtype)
    return rechits, simclusters, hitless, fallback


def to_table(converted: list[tuple[dict, dict]], event_ids: list[int], set_name: str, optional: tuple[str, ...] = (), sectors=None):
    """Arrow table of a batch of converted events: one row per event (or per sector, `sectors` = [(sector index,
    has_leading)] per row), every collection a struct of lists."""

    def jagged(arrays):
        # concatenate + unflatten keeps the dtypes (an ak.Array built from a list of arrays promotes to float64)
        return ak.unflatten(np.concatenate(arrays), np.array([len(a) for a in arrays], dtype=np.int64))

    simcluster_branches = SIMCLUSTER_BRANCHES + optional + (SECTOR_SIMCLUSTER_BRANCHES if sectors is not None else ())
    rechits = ak.zip({b: jagged([c[0][b] for c in converted]) for b in RECHIT_BRANCHES}, depth_limit=1)
    simclusters = ak.zip({b: jagged([c[1][b] for c in converted]) for b in simcluster_branches}, depth_limit=1)
    top = {
        "RecHitHGC": rechits,
        "SimCluster": simclusters,
        "event_id": np.asarray(event_ids, dtype=np.int64),
        "set_name": ak.Array([set_name] * len(event_ids)),
    }
    if sectors is not None:
        top["sector"] = np.asarray([s for s, _ in sectors], dtype=np.int32)
        top["sector_has_leading"] = np.asarray([lead for _, lead in sectors], dtype=bool)
    events = ak.zip(top, depth_limit=1)
    # plain arrow types (struct of large_list), as the CMS NanoML parquet files
    return ak.to_arrow_table(events, extensionarray=False)


def struct_columns(table, collection: str, branches) -> dict:
    """{branch: (flat numpy values, per-event offsets)} of one collection of a table read back from disk."""
    struct = table.column(collection).combine_chunks()
    out = {}
    for b in branches:
        lists = struct.field(b)
        counts = lists.value_lengths().to_numpy(zero_copy_only=False)
        out[b] = (lists.flatten().to_numpy(zero_copy_only=False), np.concatenate([[0], np.cumsum(counts)]))
    return out


def bits_equal(stored: np.ndarray, expected: np.ndarray) -> bool:
    """Same dtype, same length, identical bits (float NaN/-0.0 included)."""
    if stored.dtype != expected.dtype or stored.shape != expected.shape:
        return False
    if stored.dtype.kind == "f":
        as_bits = np.uint32 if stored.itemsize == 4 else np.uint64
        return np.array_equal(stored.view(as_bits), expected.view(as_bits))
    return np.array_equal(stored, expected)


def angle_diff(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def check_file(path: Path, reader, event_ids: list[int], set_name: str, keep_hitless: bool, report: dict) -> list[str]:
    """Re-read a written file and compare it with the source events; returns the failures (empty = pass).

    Updates `report` with the largest float32 rounding (relative to the float64 source) per float branch, the
    largest eta/phi deviation from a recomputation on the stored float32 position, and, with the optional branches,
    the largest difference between _truthAssistedEnergy and the summed stored energy of the particle's hits."""
    failures = []
    pf = pq.ParquetFile(path)
    if pf.metadata.num_rows != len(event_ids):
        return [f"{path.name}: {pf.metadata.num_rows} rows, expected {len(event_ids)}"]
    optional = optional_branches(reader.get_element(event_ids[0]))
    expected = {"RecHitHGC": RECHIT_BRANCHES, "SimCluster": SIMCLUSTER_BRANCHES + optional}
    for collection, branches in expected.items():
        stored = tuple(f.name for f in pf.schema_arrow.field(collection).type)
        if stored != branches:
            return [f"{path.name}: {collection} branches {stored}, expected {branches}"]
    rounding = report.setdefault("float32_max_rel_rounding", {})
    recompute = report.setdefault("eta_phi_max_dev_from_float32_position", {"eta": 0.0, "phi": 0.0})
    geometry = report.setdefault("impact_point_geometry", {"face_max_dev_cm": 0.0, "line_max_dev_cm": 0.0})
    if "SimCluster_truthAssistedEnergy" in optional:
        hit_sum = report.setdefault("truth_assisted_vs_hit_sum", {"max_abs_dev_gev": 0.0, "max_rel_dev": 0.0})
    first = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        rh = struct_columns(table, "RecHitHGC", RECHIT_BRANCHES)
        sc = struct_columns(table, "SimCluster", SIMCLUSTER_BRANCHES + optional)
        ids = table.column("event_id").to_numpy()
        names = table.column("set_name").to_pylist()
        for j in range(table.num_rows):
            eid = event_ids[first + j]
            tag = f"{path.name} event_id {eid}"
            if ids[j] != eid or names[j] != set_name:
                failures.append(f"{tag}: top-level event_id/set_name = {ids[j]}/{names[j]}")
            ev = reader.get_element(eid)
            got = {b: v[o[j]:o[j + 1]] for b, (v, o) in rh.items()}
            got.update({b: v[o[j]:o[j + 1]] for b, (v, o) in sc.items()})
            truth = np.asarray(ev["truth_assignment"])
            p = ev["particles"]
            rows = simcluster_rows(ev, keep_hitless)

            def expect(branch, source, dtype):
                source = np.asarray(source)
                want = source.astype(dtype)
                if not bits_equal(got[branch], want):
                    failures.append(f"{tag}: {branch} differs from {dtype.__name__}(source)")
                elif want.dtype == np.float32 and source.dtype.kind == "f" and source.size:
                    rel = np.abs(want.astype(np.float64) - source) / np.maximum(np.abs(source), np.finfo(np.float64).tiny)
                    rounding[branch] = max(rounding.get(branch, 0.0), float(rel.max()))

            # per hit: copied branches
            expect("RecHitHGC_x", ev["rechit_x"], np.float32)
            expect("RecHitHGC_y", ev["rechit_y"], np.float32)
            expect("RecHitHGC_z", ev["rechit_z"], np.float32)
            expect("RecHitHGC_energy", ev["rechit_energy"], np.float32)
            expect("RecHitHGC_area", ev["rechit_area"], np.float32)
            expect("RecHitHGC_detId", ev["rechit_sensor_idx"], np.int32)
            expect("RecHitHGC_layer", ev["rechit_layer"], np.int32)
            expect("RecHitHGC_time", np.full(len(truth), TIME_SENTINEL), np.float32)
            expect("RecHitHGC_SimClusterBestMatchQual", truth >= 0, np.float32)
            # per hit: the match index points at the SimCluster whose table row is truth_assignment
            idx, calo = got["RecHitHGC_SimClusterBestMatchIdx"], got["SimCluster_CaloPartIdx"]
            if idx.size and (idx.max() >= len(calo) or idx.min() < -1):
                failures.append(f"{tag}: BestMatchIdx outside [-1, {len(calo) - 1}]")
            elif not np.array_equal(np.where(idx >= 0, calo[np.maximum(idx, 0)], -1), truth):
                failures.append(f"{tag}: CaloPartIdx[BestMatchIdx] != truth_assignment")

            # per particle: the rows, then the copied and derived branches
            expect("SimCluster_CaloPartIdx", rows, np.int32)
            expect("SimCluster_trackIdAtBoundary", rows, np.int32)
            if not keep_hitless and not np.array_equal(np.unique(got["SimCluster_CaloPartIdx"]), np.unique(truth[truth >= 0])):
                failures.append(f"{tag}: SimClusters are not exactly the particles with >= 1 rechit")
            face, recorded, fallback = impact_points(p, rows)
            for k, c in enumerate("xyz"):
                expect(f"SimCluster_impactPoint_{c}", face[:, k] / 10.0, np.float32)
                expect(f"SimCluster_firstImpact_{c}", recorded[:, k] / 10.0, np.float32)
            eta, phi = impact_eta_phi(face[:, 0] / 10.0, face[:, 1] / 10.0, face[:, 2] / 10.0)
            expect("SimCluster_impactPoint_eta", eta, np.float32)
            expect("SimCluster_impactPoint_phi", phi, np.float32)
            # independent of impact_points(): a stored crossing lies on the front face and on the line through the
            # source vertex along the source direction (cross-product distance), a fallback is the recorded point
            stored = np.stack([got[f"SimCluster_impactPoint_{c}"].astype(np.float64) for c in "xyz"], axis=1)
            vertex = np.stack([np.asarray(p[f"particles_vertex_position_{c}"], dtype=np.float64)[rows] / 10.0 for c in "xyz"], axis=1)
            direction = np.stack([np.asarray(p[f"particles_momentum_direction_{c}"], dtype=np.float64)[rows] for c in "xyz"], axis=1)
            crossing = (direction[:, 2] > 0) & (vertex[:, 2] < CALO_FRONT_Z / 10.0)
            if crossing.any():
                d_face = float(np.abs(stored[crossing, 2] - CALO_FRONT_Z / 10.0).max())
                d_line = float((np.linalg.norm(np.cross(stored - vertex, direction), axis=1) / np.linalg.norm(direction, axis=1))[crossing].max())
                geometry["face_max_dev_cm"] = max(geometry["face_max_dev_cm"], d_face)
                geometry["line_max_dev_cm"] = max(geometry["line_max_dev_cm"], d_line)
                if d_face > FACE_TOL_CM or d_line > LINE_TOL_CM:
                    failures.append(f"{tag}: impact point {d_face:.2e} cm off the front face / {d_line:.2e} cm off the particle line")
            first32 = np.stack([got[f"SimCluster_firstImpact_{c}"].astype(np.float64) for c in "xyz"], axis=1)
            if not np.array_equal(stored[~crossing], first32[~crossing]):
                failures.append(f"{tag}: a fallback impact point differs from the recorded first impact")
            expect("SimCluster_recEnergy", np.asarray(p["particles_kinetic_energy"])[rows], np.float32)
            expect("SimCluster_boundaryEnergy", np.asarray(p["particles_kinetic_energy"])[rows], np.float32)
            expect("SimCluster_pdgId", np.asarray(p["particles_pdgid"])[rows], np.int32)
            expect("SimCluster_eventId", np.asarray(p["particles_only_minbias"], dtype=bool)[rows], np.int32)
            expect("SimCluster_isPileup", np.asarray(p["particles_only_minbias"], dtype=bool)[rows], np.bool_)
            expect("SimCluster_merged", np.asarray(p["merged"], dtype=bool)[rows], np.bool_)
            expect("SimCluster_bunchCrossing", np.zeros(len(rows)), np.int32)
            # optional branches: the builder's values, and consistency with the stored hits matched to each SimCluster
            for branch in optional:
                column, dtype = OPTIONAL_SIMCLUSTER_BRANCHES[branch]
                expect(branch, np.asarray(p[column])[rows], dtype)
            matched = idx >= 0
            if "SimCluster_nHits" in optional:
                counted = np.bincount(idx[matched], minlength=len(rows))
                if not np.array_equal(counted, got["SimCluster_nHits"]):
                    failures.append(f"{tag}: SimCluster_nHits differs from the number of hits matched to each SimCluster")
            if "SimCluster_truthAssistedEnergy" in optional and len(rows):
                summed = np.bincount(idx[matched], weights=got["RecHitHGC_energy"][matched].astype(np.float64), minlength=len(rows))
                stored_sum = got["SimCluster_truthAssistedEnergy"].astype(np.float64)
                d_abs = np.abs(summed - stored_sum)
                d_rel = float((d_abs / np.maximum(np.abs(stored_sum), np.finfo(np.float64).tiny))[stored_sum > 0].max(initial=0.0))
                hit_sum["max_abs_dev_gev"] = max(hit_sum["max_abs_dev_gev"], float(d_abs.max()))
                hit_sum["max_rel_dev"] = max(hit_sum["max_rel_dev"], d_rel)
                if d_rel > TRUTH_SUM_REL_TOL or (np.any(stored_sum == 0) and d_abs[stored_sum == 0].max() > 0):
                    failures.append(f"{tag}: SimCluster_truthAssistedEnergy differs from the summed hit energy by {d_rel:.2e} (relative)")
            # eta/phi as a reader recomputes them from the stored float32 position
            eta32, phi32 = impact_eta_phi(*(got[f"SimCluster_impactPoint_{c}"].astype(np.float64) for c in "xyz"))
            if len(rows):
                d_eta = float(np.abs(eta32 - got["SimCluster_impactPoint_eta"]).max())
                d_phi = float(angle_diff(phi32, got["SimCluster_impactPoint_phi"].astype(np.float64)).max())
                recompute["eta"] = max(recompute["eta"], d_eta)
                recompute["phi"] = max(recompute["phi"], d_phi)
                if d_eta > ETA_PHI_TOL or d_phi > ETA_PHI_TOL:
                    failures.append(f"{tag}: eta/phi differ from the stored position by {d_eta:.2e}/{d_phi:.2e} > {ETA_PHI_TOL}")
        first += table.num_rows
    return failures


def check_sector_file(path: Path, reader, event_ids: list[int], set_name: str, n_sectors: int, offset_deg: float,
                      target: str, report: dict, min_share: float = 0.0) -> list[str]:
    """Re-read a sector-event file and compare each event's sector rows with the event's full conversion
    (convert_event, itself validated against the source by check_file); returns the failures (empty = pass).

    Per event: the rows are sectors 0..n-1; every hit appears in exactly one row (matched by sensor index) with its
    branches bit-identical to the full conversion, and the sector recomputed from its stored float32 position is the
    row's, boundary sensors included (SECTOR_TIE_RAD). A hit's BestMatchIdx points at its own particle, or is -1 (with
    BestMatchQual 0) for noise and for the hits of a particle below the share cut. The SimClusters of a row are exactly
    the particles whose share of hit energy in the sector, recomputed from the stored hits, is >= min_share (shares
    within TRUTH_SUM_REL_TOL of the cut may go either way); their branches equal the full conversion's (recEnergy too,
    unless target == "deposited", where it is the matched stored hit energy), _fullEnergy is the full true energy,
    _sectorNHits counts the matched hits and _sectorEnergyFraction is the recomputed share (relative tolerance). With
    min_share <= 1/n every particle is a target in at least one sector. One row per event has sector_has_leading: the
    sector with the largest stored hit energy of the most energetic SimCluster."""
    failures = []
    pf = pq.ParquetFile(path)
    if pf.metadata.num_rows != n_sectors * len(event_ids):
        return [f"{path.name}: {pf.metadata.num_rows} rows, expected {n_sectors} x {len(event_ids)}"]
    optional = optional_branches(reader.get_element(event_ids[0]))
    simcluster_branches = SIMCLUSTER_BRANCHES + optional + SECTOR_SIMCLUSTER_BRANCHES
    expected = {"RecHitHGC": RECHIT_BRANCHES, "SimCluster": simcluster_branches}
    for collection, branches in expected.items():
        stored = tuple(f.name for f in pf.schema_arrow.field(collection).type)
        if stored != branches:
            return [f"{path.name}: {collection} branches {stored}, expected {branches}"]
    top = ["RecHitHGC", "SimCluster", "event_id", "set_name", "sector", "sector_has_leading"]
    if pf.schema_arrow.names != top:
        return [f"{path.name}: top-level columns {pf.schema_arrow.names}, expected {top}"]
    stats = report.setdefault("sector_check", {"hits_on_a_boundary": 0, "fraction_max_dev": 0.0, "deposited_max_rel_dev": 0.0,
                                               "entries_below_cut": 0, "entries_at_cut": 0})
    boundaries = np.radians(offset_deg + np.arange(n_sectors) * 360.0 / n_sectors)
    tiny = np.finfo(np.float64).tiny
    first = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        if table.num_rows % n_sectors:
            return failures + [f"{path.name}: row group {rg} holds {table.num_rows} rows, not whole events of {n_sectors} sectors"]
        rh = struct_columns(table, "RecHitHGC", RECHIT_BRANCHES)
        sc = struct_columns(table, "SimCluster", simcluster_branches)
        ids, names = table.column("event_id").to_numpy(), table.column("set_name").to_pylist()
        sectors, leads = table.column("sector").to_numpy(), table.column("sector_has_leading").to_numpy(zero_copy_only=False)
        for e in range(table.num_rows // n_sectors):
            eid = event_ids[first // n_sectors + e]
            tag = f"{path.name} event_id {eid}"
            ev = reader.get_element(eid)
            full_rh, full_sc, _, _ = convert_event(ev, False, optional)
            truth = np.asarray(ev["truth_assignment"])
            sensor = full_rh["RecHitHGC_detId"]
            order = np.argsort(sensor)
            full_rows = full_sc["SimCluster_CaloPartIdx"].astype(np.int64)
            n_table = int(full_rows.max(initial=-1)) + 1
            energy_by_sector = np.zeros((n_sectors, n_table))  # stored hit energy per particle (by truth), every hit
            rows_got, seen = [], []
            # pass 1: per row, everything that does not need the event's totals
            for k in range(n_sectors):
                j = e * n_sectors + k
                if ids[j] != eid or names[j] != set_name or sectors[j] != k:
                    failures.append(f"{tag}: row {j} has event_id/set_name/sector {ids[j]}/{names[j]}/{sectors[j]}, expected sector {k}")
                    rows_got.append(None)
                    continue
                got = {b: v[o[j]:o[j + 1]] for b, (v, o) in rh.items()}
                got.update({b: v[o[j]:o[j + 1]] for b, (v, o) in sc.items()})
                det = got["RecHitHGC_detId"]
                seen.append(det)
                at = np.searchsorted(sensor[order], det)
                if np.any(at >= len(sensor)) or not np.array_equal(sensor[order][np.minimum(at, len(sensor) - 1)], det):
                    failures.append(f"{tag} sector {k}: a stored hit's sensor is not a hit of the event")
                    rows_got.append(None)
                    continue
                src = order[at]  # index of each stored hit in the event
                for b in RECHIT_BRANCHES:
                    if b not in ("RecHitHGC_SimClusterBestMatchIdx", "RecHitHGC_SimClusterBestMatchQual") and not bits_equal(got[b], full_rh[b][src]):
                        failures.append(f"{tag} sector {k}: {b} differs from the full conversion")
                x32, y32 = got["RecHitHGC_x"].astype(np.float64), got["RecHitHGC_y"].astype(np.float64)
                off = sector_of(x32, y32, n_sectors, offset_deg) != k
                if off.any():
                    failures.append(f"{tag} sector {k}: {int(off.sum())} hits belong to another sector by their stored position")
                if len(x32):
                    on_boundary = np.min(angle_diff(np.arctan2(y32, x32)[:, None], boundaries[None, :]), axis=1) < 1e-7
                    stats["hits_on_a_boundary"] += int(on_boundary.sum())
                idx, calo = got["RecHitHGC_SimClusterBestMatchIdx"].astype(np.int64), got["SimCluster_CaloPartIdx"].astype(np.int64)
                if idx.size and (idx.max() >= len(calo) or idx.min() < -1):
                    failures.append(f"{tag} sector {k}: BestMatchIdx outside [-1, {len(calo) - 1}]")
                    rows_got.append(None)
                    continue
                t_src = truth[src]
                mapped = np.where(idx >= 0, calo[np.maximum(idx, 0)] if len(calo) else -1, -1)
                # a hit maps to its own particle, or is noise: true noise, or a particle that is not a target here
                if not np.all((mapped == t_src) | ((mapped == -1) & ~np.isin(t_src, calo))):
                    failures.append(f"{tag} sector {k}: BestMatchIdx maps hits to the wrong particle")
                if not bits_equal(got["RecHitHGC_SimClusterBestMatchQual"], (idx >= 0).astype(np.float32)):
                    failures.append(f"{tag} sector {k}: BestMatchQual is not 1 exactly for the matched hits")
                if not (np.array_equal(calo, np.unique(calo)) and np.all(np.isin(calo, t_src))):
                    failures.append(f"{tag} sector {k}: SimClusters are not sorted particles with hits in the sector")
                    rows_got.append(None)
                    continue
                pos = np.searchsorted(full_rows, calo)
                for b in SIMCLUSTER_BRANCHES + optional:
                    if (b != "SimCluster_recEnergy" or target == "full") and not bits_equal(got[b], full_sc[b][pos]):
                        failures.append(f"{tag} sector {k}: {b} differs from the full conversion")
                if not bits_equal(got["SimCluster_fullEnergy"], full_sc["SimCluster_boundaryEnergy"][pos]):
                    failures.append(f"{tag} sector {k}: SimCluster_fullEnergy is not the full true energy")
                matched = idx >= 0
                counted = np.bincount(idx[matched], minlength=len(calo))
                summed = np.bincount(idx[matched], weights=got["RecHitHGC_energy"][matched].astype(np.float64), minlength=len(calo))
                if not np.array_equal(counted, got["SimCluster_sectorNHits"]):
                    failures.append(f"{tag} sector {k}: SimCluster_sectorNHits differs from the hits matched in the sector")
                if target == "deposited" and len(calo):
                    rel = float((np.abs(got["SimCluster_recEnergy"].astype(np.float64) - summed) / np.maximum(summed, tiny)).max())
                    stats["deposited_max_rel_dev"] = max(stats["deposited_max_rel_dev"], rel)
                    if rel > TRUTH_SUM_REL_TOL:
                        failures.append(f"{tag} sector {k}: in-sector recEnergy differs from the summed hit energy by {rel:.2e}")
                has_truth = t_src >= 0
                energy_by_sector[k] = np.bincount(t_src[has_truth], weights=got["RecHitHGC_energy"][has_truth].astype(np.float64), minlength=n_table)
                rows_got.append((calo, got["SimCluster_sectorEnergyFraction"].astype(np.float64), np.unique(t_src[has_truth])))
            if len(seen) == n_sectors and not np.array_equal(np.sort(np.concatenate(seen)), np.sort(sensor)):
                failures.append(f"{tag}: the sector rows do not hold every hit of the event exactly once")
            if any(r is None for r in rows_got):
                continue
            # pass 2: the shares, recomputed from the stored hits, decide the targets and the stored fractions
            total = energy_by_sector.sum(axis=0)
            targets_anywhere = np.zeros(n_table, dtype=bool)
            for k, (calo, frac, candidates) in enumerate(rows_got):
                share = energy_by_sector[k, candidates] / np.maximum(total[candidates], tiny)
                at_cut = np.abs(share - min_share) <= min_share * TRUTH_SUM_REL_TOL
                must, may = candidates[(share >= min_share) & ~at_cut], candidates[at_cut]
                stats["entries_at_cut"] += int(at_cut.sum())
                stats["entries_below_cut"] += int(np.sum((share < min_share) & ~at_cut))
                if not (np.all(np.isin(must, calo)) and np.all(np.isin(calo, np.concatenate([must, may])))):
                    failures.append(f"{tag} sector {k}: the SimClusters are not the particles with >= {min_share:g} of their hit energy here")
                targets_anywhere[calo] = True
                if len(calo):
                    want = energy_by_sector[k, calo] / np.maximum(total[calo], tiny)
                    dev = float((np.abs(frac - want) / np.maximum(want, tiny)).max())
                    stats["fraction_max_dev"] = max(stats["fraction_max_dev"], dev)
                    if dev > TRUTH_SUM_REL_TOL:
                        failures.append(f"{tag} sector {k}: _sectorEnergyFraction differs from the stored hit energies by {dev:.2e} (relative)")
            hit_rows = np.unique(truth[truth >= 0])
            if min_share <= 1.0 / n_sectors and not np.all(targets_anywhere[hit_rows]):
                failures.append(f"{tag}: {int(np.sum(~targets_anywhere[hit_rows]))} particles are a target in no sector")
            flagged = np.flatnonzero(leads[e * n_sectors:(e + 1) * n_sectors])
            leading = full_rows[np.argmax(full_sc["SimCluster_boundaryEnergy"])] if len(full_rows) else -1
            # the converter picks the sector on float64 source energies, this check sums stored float32 ones: a near-tie
            # within TRUTH_SUM_REL_TOL passes
            if leading >= 0:
                lead_energy = energy_by_sector[:, leading]
                if len(flagged) != 1 or lead_energy[flagged[0]] < lead_energy.max() * (1 - TRUTH_SUM_REL_TOL):
                    failures.append(f"{tag}: sector_has_leading {flagged.tolist()} is not the sector with most of the leading particle's hit energy")
        first += table.num_rows
    return failures


def multiplicity(n) -> dict:
    n = np.asarray(n, dtype=np.float64)
    return {"mean": float(n.mean()), "std": float(n.std()), "min": float(n.min()), "max": float(n.max())} if n.size else {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="ra_pickles directory of one event set (toy event builder output)")
    p.add_argument("--output-dir", required=True, help="Directory for the parquet files (must not contain any yet)")
    p.add_argument("--set-name", default=None, help="Set name, used in the file names and the set_name column (default: input directory name)")
    p.add_argument("--events-per-file", type=int, default=1000, help="Events per parquet file (the CMS files hold 250-1,000)")
    p.add_argument("--row-group-events", type=int, default=100, help="Events per row group (bounds the memory of streaming readers)")
    p.add_argument("--max-events", type=int, default=None, help="Convert only the first N events (smoke tests)")
    p.add_argument("--keep-hitless-particles", action="store_true", help="Keep particle-table rows without any rechit as SimClusters")
    p.add_argument("--compression", default="lz4", help="Parquet compression codec")
    p.add_argument("--no-check", action="store_true", help="Skip the re-read comparison with the source events")
    p.add_argument("--sectors", type=int, default=0, help="Write every event as this many azimuthal sector events (0 = off; 3 = 120-degree sectors)")
    p.add_argument("--sector-offset-deg", type=float, default=0.0, help="Azimuth of the first sector boundary in degrees (e.g. 60 for a rotated set)")
    p.add_argument("--sector-target", choices=("full", "deposited"), default="full",
                   help="Sector events: SimCluster_recEnergy = the particle's full true energy, or its hit energy in the sector")
    p.add_argument("--sector-min-share", type=float, default=0.0,
                   help="Sector events: a particle is a SimCluster of a sector only with >= this share of its hit energy there; "
                   "its hits in a sector below the cut are noise (default 0: every particle with >= 1 hit)")
    args = p.parse_args(argv)
    if args.sectors == 1 or args.sectors < 0:
        p.error("--sectors must be 0 (off) or >= 2")
    if not 0.0 <= args.sector_min_share < 1.0:
        p.error("--sector-min-share must be in [0, 1)")
    if args.sector_min_share and not args.sectors:
        p.error("--sector-min-share needs --sectors")
    if args.sectors and args.keep_hitless_particles:
        p.error("--sectors keeps the particles with >= 1 hit in each sector; it does not combine with --keep-hitless-particles")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    in_dir, out_dir = Path(args.input), Path(args.output_dir)
    set_name = args.set_name or in_dir.name
    if out_dir.exists() and any(out_dir.glob("*.parquet*")):
        print(f"ERROR: {out_dir} already holds parquet files; refusing to overwrite", file=sys.stderr)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = ra_pickles.RandomAccessPicklesReader(str(in_dir))
    n_total = reader.get_total()
    n_events = n_total if args.max_events is None else min(n_total, args.max_events)
    # optional SimCluster branches: decided once per set, from its first event
    optional = optional_branches(reader.get_element(0))
    print(f"optional SimCluster branches written: {list(optional) or 'none (the event builder provides no source column)'}")
    print(f"Converting {n_events}/{n_total} events of {in_dir} -> {out_dir} (set {set_name}, {args.events_per_file} per file, "
          f"row groups of {args.row_group_events}, {'every particle-table row' if args.keep_hitless_particles else 'particles with >= 1 rechit'})",
          flush=True)
    if args.sectors:
        print(f"SECTOR EVENTS: {args.sectors} sectors of {360 / args.sectors:g} deg per event, first boundary at {args.sector_offset_deg:g} deg, "
              f"SimCluster_recEnergy = {'full true energy' if args.sector_target == 'full' else 'in-sector hit energy'}, "
              f"target if >= {args.sector_min_share:g} of the particle's hit energy is in the sector", flush=True)
    sector_stats = {"hits": [], "simclusters": [], "n_sectors_of_particle": [], "split_shares": [], "split_max_share": [],
                    "true_energy": [], "split_true_energy": [], "dropped_energy": 0.0, "particle_energy": 0.0}

    stats = {"hits": [], "noise_hits": [], "simclusters": [], "pileup_simclusters": [], "merged_simclusters": [], "table_rows": [], "hitless_rows": [],
             "impact_fallbacks": [], "recorded_impact_in_front": []}
    pdg_kept, pdg_hitless = Counter(), Counter()
    files, failures, check_report = [], [], {}
    for first in range(0, n_events, args.events_per_file):
        ids = list(range(first, min(first + args.events_per_file, n_events)))
        name = f"{set_name}_{first // args.events_per_file:03d}.parquet"
        partial = out_dir / (name + ".partial")
        writer = None
        for start in range(0, len(ids), args.row_group_events):
            batch_ids = ids[start:start + args.row_group_events]
            converted, row_ids, row_sectors = [], [], []
            for eid in batch_ids:
                ev = reader.get_element(eid)
                rechits, simclusters, hitless, fallback = convert_event(ev, args.keep_hitless_particles, optional)
                if args.sectors:
                    split, dropped_energy, particle_energy = split_into_sectors(ev, rechits, simclusters, args.sectors, args.sector_offset_deg,
                                                                               args.sector_target, args.sector_min_share)
                    sector_stats["dropped_energy"] += dropped_energy
                    sector_stats["particle_energy"] += particle_energy
                    shares = {}
                    for k, (sector_rechits, sector_simclusters, has_leading) in enumerate(split):
                        converted.append((sector_rechits, sector_simclusters))
                        row_ids.append(eid)
                        row_sectors.append((k, has_leading))
                        sector_stats["hits"].append(len(sector_rechits["RecHitHGC_x"]))
                        sector_stats["simclusters"].append(len(sector_simclusters["SimCluster_pdgId"]))
                        for row, share in zip(sector_simclusters["SimCluster_CaloPartIdx"].tolist(), sector_simclusters["SimCluster_sectorEnergyFraction"].tolist()):
                            shares.setdefault(row, []).append(share)
                    true_energy = dict(zip(simclusters["SimCluster_CaloPartIdx"].tolist(), simclusters["SimCluster_boundaryEnergy"].tolist()))
                    for row, s in shares.items():
                        sector_stats["n_sectors_of_particle"].append(len(s))
                        sector_stats["true_energy"].append(true_energy[row])
                        if len(s) > 1:
                            sector_stats["split_shares"] += s
                            sector_stats["split_max_share"].append(max(s))
                            sector_stats["split_true_energy"].append(true_energy[row])
                else:
                    converted.append((rechits, simclusters))
                    row_ids.append(eid)
                stats["hits"].append(len(rechits["RecHitHGC_x"]))
                stats["noise_hits"].append(int((rechits["RecHitHGC_SimClusterBestMatchIdx"] == -1).sum()))
                stats["simclusters"].append(len(simclusters["SimCluster_pdgId"]))
                stats["pileup_simclusters"].append(int(simclusters["SimCluster_isPileup"].sum()))
                stats["merged_simclusters"].append(int(simclusters["SimCluster_merged"].sum()))
                stats["table_rows"].append(len(ev["particles"]["particles_kinetic_energy"]))
                stats["hitless_rows"].append(len(hitless))
                stats["impact_fallbacks"].append(int(fallback.sum()))
                # recorded first impact more than 1 mm in front of the face (a charged particle's first world step)
                stats["recorded_impact_in_front"].append(int((simclusters["SimCluster_firstImpact_z"] < CALO_FRONT_Z / 10.0 - 0.1).sum()))
                pdg_kept.update(simclusters["SimCluster_pdgId"].tolist())
                pdg_hitless.update(np.asarray(ev["particles"]["particles_pdgid"])[hitless].tolist())
            table = to_table(converted, row_ids, set_name, optional, row_sectors if args.sectors else None)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression=args.compression)
            writer.write_table(table)
            del converted, table
        writer.close()

        if args.no_check:
            file_failures = []
        elif args.sectors:
            file_failures = check_sector_file(partial, reader, ids, set_name, args.sectors, args.sector_offset_deg, args.sector_target, check_report,
                                              args.sector_min_share)
        else:
            file_failures = check_file(partial, reader, ids, set_name, args.keep_hitless_particles, check_report)
        if file_failures:
            failures += file_failures
            print(f"CHECK FAILED for {partial} ({len(file_failures)} failures), kept under its .partial name; first: {file_failures[:5]}", flush=True)
            break
        final = out_dir / name
        os.replace(partial, final)
        files.append({"name": name, "events": len(ids), "bytes": final.stat().st_size, **({"rows": len(ids) * args.sectors} if args.sectors else {})})
        print(f"  wrote {final} ({len(ids)} events, {final.stat().st_size / 1e6:.1f} MB){'' if args.no_check else ', check PASSED'}"
              f"  [{time.time() - t_start:.0f} s]", flush=True)

    hits, noise = np.array(stats["hits"]), np.array(stats["noise_hits"])
    summary = {
        "input": str(in_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "set_name": set_name,
        "events_in_input": n_total,
        "events_converted": len(hits),
        "keep_hitless_particles": args.keep_hitless_particles,
        "optional_simcluster_branches": list(optional),
        "files": files,
        "hits_per_event": multiplicity(hits),
        "noise_fraction": float(noise.sum() / hits.sum()) if hits.sum() else 0.0,
        "simclusters_per_event": multiplicity(stats["simclusters"]),
        "pileup_simclusters_per_event": multiplicity(stats["pileup_simclusters"]),
        "merged_simclusters_per_event": multiplicity(stats["merged_simclusters"]),
        "particle_table_rows_per_event": multiplicity(stats["table_rows"]),
        "hitless_rows_per_event": multiplicity(stats["hitless_rows"]),
        "impact_point_fallbacks": int(np.sum(stats["impact_fallbacks"])),
        "recorded_impact_in_front_of_face": int(np.sum(stats["recorded_impact_in_front"])),
        "simclusters_total": int(np.sum(stats["simclusters"])),
        "pdgId_simclusters": dict(sorted(pdg_kept.items())),
        "pdgId_hitless_rows": dict(sorted(pdg_hitless.items())),
        "check": "skipped" if args.no_check else ("FAILED" if failures else "PASSED"),
        "check_failures": failures[:50],
        "check_report": check_report,
        "eta_phi_tol": ETA_PHI_TOL,
        "seconds": round(time.time() - t_start, 1),
    }
    if args.sectors:
        n_sec = np.array(sector_stats["n_sectors_of_particle"])
        split_e, all_e = np.sum(sector_stats["split_true_energy"]), np.sum(sector_stats["true_energy"])
        summary["sectors"] = {
            "n_sectors": args.sectors,
            "offset_deg": args.sector_offset_deg,
            "target": args.sector_target,
            "min_share": args.sector_min_share,
            "particle_hit_energy_relabelled_noise_fraction": sector_stats["dropped_energy"] / sector_stats["particle_energy"] if sector_stats["particle_energy"] else 0.0,
            "rows": len(sector_stats["hits"]),
            "hits_per_sector": multiplicity(sector_stats["hits"]),
            "simclusters_per_sector": multiplicity(sector_stats["simclusters"]),
            "sectors_without_simclusters": int(np.sum(np.array(sector_stats["simclusters"]) == 0)),
            "particles": int(n_sec.size),
            "particles_in_n_sectors": {str(n): int(np.sum(n_sec == n)) for n in range(1, args.sectors + 1)},
            "split_particle_fraction": float(np.mean(n_sec > 1)) if n_sec.size else 0.0,
            "split_true_energy_fraction": float(split_e / all_e) if all_e else 0.0,
            "split_share_quantiles_5_25_50_75_95": [float(q) for q in np.quantile(sector_stats["split_shares"], [0.05, 0.25, 0.5, 0.75, 0.95])] if sector_stats["split_shares"] else [],
            "split_max_share_quantiles_5_25_50_75_95": [float(q) for q in np.quantile(sector_stats["split_max_share"], [0.05, 0.25, 0.5, 0.75, 0.95])] if sector_stats["split_max_share"] else [],
            "split_share_histogram_0.1": np.histogram(sector_stats["split_shares"], bins=np.linspace(0, 1, 11))[0].tolist(),
        }
    (out_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    h = summary["hits_per_event"]
    s = summary["simclusters_per_event"]
    print(f"\nset {set_name}: {summary['events_converted']} events in {len(files)} files; hits/event {h['mean']:.1f} +- {h['std']:.1f}; "
          f"SimClusters/event {s['mean']:.2f} +- {s['std']:.2f} (pileup {summary['pileup_simclusters_per_event']['mean']:.2f}); "
          f"noise fraction {summary['noise_fraction']:.4f}; hit-less table rows/event {summary['hitless_rows_per_event']['mean']:.2f} "
          f"{'kept' if args.keep_hitless_particles else 'dropped'} (pdgIds {summary['pdgId_hitless_rows']})")
    rounding = check_report.get("float32_max_rel_rounding", {})
    print(f"check: {summary['check']}" + ("; float32 rounding (max relative): " + ", ".join(f"{k.split('_', 1)[1]} {v:.2e}" for k, v in rounding.items())
                                         if rounding else " (sector rows compared with the full conversion)"))
    if "eta_phi_max_dev_from_float32_position" in check_report:
        d = check_report["eta_phi_max_dev_from_float32_position"]
        print(f"eta/phi vs recomputation from the stored float32 position: max |d eta| {d['eta']:.2e}, max |d phi| {d['phi']:.2e} (tolerance {ETA_PHI_TOL:g})")
        g = check_report["impact_point_geometry"]
        print(f"impact point (front-face crossing): max {g['face_max_dev_cm']:.1e} cm off the face, {g['line_max_dev_cm']:.1e} cm off the particle line "
              f"(tolerances {FACE_TOL_CM:g} / {LINE_TOL_CM:g} cm)")
        if "truth_assisted_vs_hit_sum" in check_report:
            t = check_report["truth_assisted_vs_hit_sum"]
            print(f"SimCluster_truthAssistedEnergy vs summed stored hit energy: max |d| {t['max_abs_dev_gev']:.2e} GeV, "
                  f"max relative {t['max_rel_dev']:.2e} (tolerance {TRUTH_SUM_REL_TOL:g})")
    print(f"impact points: {summary['impact_point_fallbacks']} fallbacks to the recorded first impact; recorded first impact in front of the face "
          f"for {summary['recorded_impact_in_front_of_face']}/{summary['simclusters_total']} SimClusters")
    if args.sectors:
        sec = summary["sectors"]
        hs, ss = sec["hits_per_sector"], sec["simclusters_per_sector"]
        print(f"sectors: {sec['rows']} rows; hits/sector {hs['mean']:.1f} +- {hs['std']:.1f} (max {hs['max']:.0f}); SimClusters/sector "
              f"{ss['mean']:.2f} +- {ss['std']:.2f} (max {ss['max']:.0f}); rows without SimClusters {sec['sectors_without_simclusters']}")
        print(f"share cut {sec['min_share']:g}: {100 * sec['particle_hit_energy_relabelled_noise_fraction']:.3f} % of the particles' hit energy relabelled noise")
        print(f"split particles (a target in > 1 sector): {100 * sec['split_particle_fraction']:.2f} % of {sec['particles']} (by sector count {sec['particles_in_n_sectors']}), "
              f"{100 * sec['split_true_energy_fraction']:.2f} % of the true energy; their largest share q5/q25/q50/q75/q95 "
              f"{[round(q, 3) for q in sec['split_max_share_quantiles_5_25_50_75_95']]}; all their shares in 0.1 bins {sec['split_share_histogram_0.1']}")
        if "sector_check" in check_report:
            c = check_report["sector_check"]
            print(f"sector check: every hit's sector agrees with its stored position ({c['hits_on_a_boundary']} hits sit on a boundary); fraction vs stored energies max relative "
                  f"{c['fraction_max_dev']:.1e}; deposited target max rel {c['deposited_max_rel_dev']:.1e}; entries below the share cut {c['entries_below_cut']}, "
                  f"within the tolerance of the cut {c['entries_at_cut']}")
    print(f"summary -> {out_dir / 'conversion_summary.json'}  ({summary['seconds']:.0f} s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
