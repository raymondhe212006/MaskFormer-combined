"""Resolve particles seeded by a CaloParticle to a SimCluster of that CaloParticle.

TICL's SimTrackstersProducer (RecoHGCal/TICL) builds ``ticlSimTracksters`` in two ways, per
CaloParticle. If the CaloParticle's G4 track crossed the HGCAL boundary (the particle entered
the calorimeter intact: an unconverted photon, a muon, a charged hadron), one SimTrackster is
created from the CaloParticle itself and seeded by it (``seedIsCaloParticle`` True,
``seedIndex`` = CaloParticle row). Otherwise one SimTrackster is created per SimCluster of the
CaloParticle (the decay or conversion products, e.g. everything coming from a tau), seeded by
the SimCluster (``seedIndex`` = SimCluster row).

The NanoML files carry no CaloParticle table, so a CaloParticle-seeded SimTrackster cannot read
its pdgId/eventId/bunchCrossing off its seed directly. Every SimCluster does carry the row of
its CaloParticle (``SimCluster_CaloPartIdx``), so such a particle is mapped to the most
energetic SimCluster of its CaloParticle: its own SimCluster when it reached the calorimeter as
a single particle. The conversion products share the CaloParticle's eventId and bunchCrossing
and, for e/gamma, its class.

Kept free of torch so that the plotting scripts can use it too.
"""

import awkward as ak
import numpy as np


def resolve_caloparticle_seeds(seed_idx, is_caloparticle, seed_group_idx, seed_energy=None):
    """Per-particle row in the seed collection, valid for every particle.

    All arguments are jagged awkward arrays with one list per event.

    Parameters
    ----------
    seed_idx : int, per particle
        The raw seed index: a seed-collection row for SimCluster-seeded particles, a
        CaloParticle row for CaloParticle-seeded ones.
    is_caloparticle : bool, per particle
        Which particles are seeded by a CaloParticle.
    seed_group_idx : int, per seed
        CaloParticle row of every seed (``SimCluster_CaloPartIdx``).
    seed_energy : float, per seed, optional
        Picks the seed when the CaloParticle has several (the most energetic one). The first
        one in the collection is taken when omitted.

    Returns
    -------
    resolved : int64, per particle
        ``seed_idx`` for SimCluster-seeded particles, the row of the chosen seed for
        CaloParticle-seeded ones, ``-1`` for the unresolved ones.
    n_unresolved : int
        Number of CaloParticle-seeded particles without any seed of their CaloParticle in
        the event.
    """
    is_caloparticle = ak.values_astype(is_caloparticle, bool)
    if not ak.any(is_caloparticle):
        return seed_idx, 0
    local = ak.local_index(seed_group_idx, axis=1)
    key = seed_energy if seed_energy is not None else -local
    seeds = ak.zip({"group": seed_group_idx, "key": key, "index": local})
    # The particle x seed pairing below is quadratic per event: restrict it to the events that
    # hold a CaloParticle-seeded particle (the other events get an empty seed list).
    rows_cp = ak.any(is_caloparticle, axis=1)
    keep = ak.broadcast_arrays(rows_cp, seed_group_idx)[0]
    seeds = seeds[keep]
    pairs = ak.cartesian({"p": seed_idx, "s": seeds}, axis=1, nested=True)  # event * particle * seed
    match = pairs["p"] == pairs["s"]["group"]
    n_match = ak.sum(match, axis=2)
    key = ak.where(match, pairs["s"]["key"], -np.inf)
    best = ak.argmax(key, axis=2, keepdims=True)
    best_index = ak.fill_none(ak.firsts(pairs["s"]["index"][best], axis=2), -1)
    unresolved = is_caloparticle & (n_match == 0)
    resolved = ak.where(is_caloparticle, best_index, seed_idx)
    resolved = ak.where(unresolved, -1, resolved)
    return ak.values_astype(resolved, np.int64), int(ak.sum(unresolved))
