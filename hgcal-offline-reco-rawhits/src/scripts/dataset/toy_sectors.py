"""Azimuthal sector events of the toy calorimeter (ruling M2e, 2026-10-21): the sector rule shared by the parquet
converter (toy_to_parquet.py --sectors) and any reader that needs it. numpy only: no ra_pickles, torch or awkward.

A hit belongs to sector k if the azimuth of its sensor, atan2(y, x) (a ray through the beam axis), minus the offset
lies in [k, k + 1) x 360/n_sectors degrees; a sensor centred exactly on a boundary belongs to the sector that starts
there (SECTOR_TIE_RAD). With a share cut, a particle is a target (SimCluster) of a sector only if >= min_share of its
hit energy is there; its hits in a sector below the cut are noise. The ruled spec of the sectored PU-free set:
n_sectors 3, offset 0 (and 60 for the rotated set), target "deposited", min_share 0.1.
"""

import numpy as np

# Sectors are half-open, [boundary_k, boundary_k+1): the toy grid has sensor centres exactly ON the boundaries (paper
# geometry: 3,090 at phi = 0 and 180 deg (y = 0), 957 at each of +-60 and +-120 deg), so the azimuth is nudged by this
# much (rad) before the floor. It is far above the float32 phi error (~1e-7 rad) and far below half a sensor
# (>= 0.01 rad), so the float64 source and the stored float32 position give the same sector for every sensor.
SECTOR_TIE_RAD = 1e-6
# Extra SimCluster branches of sector events (--sectors).
SECTOR_SIMCLUSTER_BRANCHES = ("SimCluster_sectorEnergyFraction", "SimCluster_sectorNHits", "SimCluster_fullEnergy")


def sector_of(x, y, n_sectors: int, offset_deg: float) -> np.ndarray:
    """Azimuthal sector of points (x, y): k such that atan2(y, x) - offset lies in [k, k + 1) x 360/n_sectors degrees,
    a point on a boundary belonging to the sector that starts there (SECTOR_TIE_RAD)."""
    phi = np.mod(np.degrees(np.arctan2(y, x) + SECTOR_TIE_RAD) - offset_deg, 360.0)
    return np.floor(phi / (360.0 / n_sectors)).astype(np.int64) % n_sectors  # % n: mod() can return 360.0 itself


def split_into_sectors(event: dict, rechits: dict, simclusters: dict, n_sectors: int, offset_deg: float, target: str,
                       min_share: float = 0.0):
    """The full conversion of one event (rechits, simclusters from convert_event) as n_sectors sector rows.

    A particle is a SimCluster of a sector if it has >= 1 hit there carrying >= min_share of its hit energy; the hits
    of a particle below the cut are noise in that sector (BestMatchIdx -1, BestMatchQual 0).
    Returns ([(rechits, simclusters, has_leading)] in sector order, the particle hit energy relabelled noise by the
    cut, the event's total particle hit energy); see the module docstring for the content."""
    truth = np.asarray(event["truth_assignment"])
    energy = np.asarray(event["rechit_energy"], dtype=np.float64)
    sector = sector_of(np.asarray(event["rechit_x"], dtype=np.float64), np.asarray(event["rechit_y"], dtype=np.float64), n_sectors, offset_deg)
    rows = simclusters["SimCluster_CaloPartIdx"].astype(np.int64)  # sorted table rows of the full event
    n_table = int(rows.max(initial=-1)) + 1
    assigned = truth >= 0
    energy_total = np.bincount(truth[assigned], weights=energy[assigned], minlength=n_table)
    # the event's most energetic SimCluster, and the sector holding the largest share of its hit energy (all its hits)
    leading = rows[np.argmax(simclusters["SimCluster_boundaryEnergy"])] if len(rows) else -1
    leading_energy = [float(energy[(sector == k) & (truth == leading)].sum()) for k in range(n_sectors)]
    leading_sector = int(np.argmax(leading_energy)) if leading >= 0 else -1
    out, dropped_energy = [], 0.0
    for k in range(n_sectors):
        in_sector = sector == k
        t = truth[in_sector]
        hit = t >= 0
        hit_energy = np.bincount(t[hit], weights=energy[in_sector][hit], minlength=n_table)
        candidates = np.unique(t[hit])
        present = candidates[hit_energy[candidates] / energy_total[candidates] >= min_share]
        kept = hit & np.isin(t, present)  # hits of particles below the share cut become noise
        dropped_energy += float(energy[in_sector][hit & ~kept].sum())
        pos = np.searchsorted(rows, present)
        if not np.array_equal(rows[pos], present):
            raise ValueError("a particle with hits in a sector is not a SimCluster of the event")
        to_local = np.full(n_table, -1, dtype=np.int64)
        to_local[present] = np.arange(len(present))
        sector_rechits = {b: v[in_sector] for b, v in rechits.items()}
        sector_rechits["RecHitHGC_SimClusterBestMatchIdx"] = np.where(kept, to_local[np.where(kept, t, 0)], -1).astype(np.int32)
        sector_rechits["RecHitHGC_SimClusterBestMatchQual"] = kept.astype(np.float32)
        sector_energy = hit_energy[present]
        sector_simclusters = {b: v[pos] for b, v in simclusters.items()}
        sector_simclusters["SimCluster_sectorEnergyFraction"] = (sector_energy / energy_total[present]).astype(np.float32)
        sector_simclusters["SimCluster_sectorNHits"] = np.bincount(t[kept], minlength=n_table)[present].astype(np.int32)
        sector_simclusters["SimCluster_fullEnergy"] = simclusters["SimCluster_boundaryEnergy"][pos].copy()
        if target == "deposited":
            sector_simclusters["SimCluster_recEnergy"] = sector_energy.astype(np.float32)
        out.append((sector_rechits, sector_simclusters, k == leading_sector))
    return out, dropped_energy, float(energy_total.sum())
