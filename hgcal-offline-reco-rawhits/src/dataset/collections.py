"""Collection knowledge shared by the dataset and the plotting scripts.

Which raw parquet branches hold the canonical topocluster/particle fields of each collection,
how the truth association between the two is stored for each (topocluster, particle) pair, and
which pairs are supported. Kept free of torch/lightning/awkward imports on purpose: the
evaluation plots (src/scripts/plot/) read prediction files with only awkward + numpy and must be
able to run in an environment without the training stack.
"""

# Raw branch name for each canonical topocluster field, per topocluster collection.
# Only fields that need a name translation are listed here; anything not provided by
# the raw collection (e.g. eta/phi/R for RecHitHGC) is derived generically in __init__.
TOPO_FIELD_MAP = {
    "RecHitHGC": {
        "x": "RecHitHGC_x",
        "y": "RecHitHGC_y",
        "z": "RecHitHGC_z",
        "energy": "RecHitHGC_energy",
        "time": "RecHitHGC_time",
    },
    "LayerCluster": {
        "x": "LayerCluster_x",
        "y": "LayerCluster_y",
        "z": "LayerCluster_z",
        "eta": "LayerCluster_eta",
        "phi": "LayerCluster_phi",
        "energy": "LayerCluster_energy",
        "nhits": "LayerCluster_nHits",
    },
    # TICLv5 CLUE3D tracksters, from the TICL NanoAOD tables (CMSSW_20_0_0_pre1). The
    # position is the energy-weighted barycenter of the trackster's layer clusters. The
    # energy INPUT is deliberately `raw_energy` (sum of the layer-cluster energies) and not
    # `regressed_energy`: the model must not depend on the TICL energy-regression DNN (it is
    # 0 for CLUE3DHigh anyway). `time` uses the same -99 "no valid time" sentinel as
    # RecHitHGC_time (~37% of tracksters on the fine-calo tau sample). `em_energy` is the
    # electromagnetic part of `raw_energy` (energy of the layer clusters in the CE-E), an
    # optional input, see OPTIONAL_TOPO_INPUTS.
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

# Optional node input features backed by a canonical topocluster field that only some
# collections provide: input name (as listed under `inputs.topocluster` and keyed in the node
# scale dict) -> canonical TOPO_FIELD_MAP field. The branch is read only when the input is
# requested, so files without it keep working for configs that do not use it.
# `e_em` (trackster raw_em_energy) is exactly 0 for ~23% of the tracksters of the fine-calo
# tau sample (no layer cluster in the CE-E) and equals raw_energy for ~66%, so its scale
# dict entry uses fn: log1p rather than the log of `e`.
OPTIONAL_TOPO_INPUTS = {
    "e_em": "em_energy",
}

# Raw branch suffix for each canonical particle field of the NanoML SimCluster-like
# collections. Identical across those collections (only the "{collection}_" prefix
# differs), so their entries in PARTICLE_FIELD_MAP are generated.
SIMCLUSTER_SUFFIX_TEMPLATES = {
    "pdgId": "pdgId",
    "eta": "impactPoint_eta",
    "phi": "impactPoint_phi",
    "x": "impactPoint_x",
    "y": "impactPoint_y",
    "z": "impactPoint_z",
    "recEnergy": "recEnergy",
    "eventId": "eventId",
    "bunchCrossing": "bunchCrossing",
    # The impact point IS the SimCluster's entry point into HGCAL (SimTrackster boundary*
    # equals SimCluster_impactPoint_* at the seed index, see PARTICLE_FIELD_MAP), so the
    # boundary targets (BOUNDARY_POSITION_FIELDS) are available for these collections too.
    # The NanoML SimCluster table carries no momentum at the boundary.
    "boundary_x": "impactPoint_x",
    "boundary_y": "impactPoint_y",
    "boundary_z": "impactPoint_z",
    "boundary_eta": "impactPoint_eta",
    "boundary_phi": "impactPoint_phi",
}

# Raw branch for each canonical particle field, per particle collection. Every collection
# must provide eta/phi/x/y/z/recEnergy here; pdgId/eventId/bunchCrossing may instead come
# from the seed lookup below.
PARTICLE_FIELD_MAP = {
    **{
        collection: {canonical: f"{collection}_{suffix}" for canonical, suffix in SIMCLUSTER_SUFFIX_TEMPLATES.items()}
        for collection in ("SimCluster", "MergedSimCluster")
    },
    # TICLv5 SimTracksters seeded by SimClusters (RecoHGCal/TICL SimTrackstersProducer,
    # CMSSW_20_0_0_pre1). The position (x/y/z/eta/phi) is the barycenter of the
    # SimTrackster's layer clusters, i.e. like-for-like with the reco-trackster barycenter
    # the model is fed. The energy TARGET (canonical key `recEnergy`, whatever the source
    # branch) is `regressed_energy`: for SimTracksters this is the seed's true energy at the
    # calorimeter boundary (SimTrackstersProducer sets it from the SimCluster's momentum at
    # the boundary), i.e. the best achievable energy reconstruction. `raw_energy` (sum of
    # the layer-cluster energies the SimTrackster collected, the analogue of
    # SimCluster_recEnergy) is NOT used as a target.
    #
    # The `boundary_*` fields describe the SEED PARTICLE where its Geant4 track crossed the
    # HGCAL front face (SimTrack::getPositionAtBoundary / getMomentumAtBoundary, dumped by
    # the TICL NanoML SimTrackster table): `boundary_{x,y,z}` is the entry point (cm), it
    # coincides with SimCluster_impactPoint_* of the seed to < 0.01 cm; `boundary_eta` is the
    # eta OF THAT POINT (equal to eta of the position vector to 1e-6, not the momentum's).
    # `boundary_phi` is BROKEN in the CMSSW_20_0_0_pre1 TICL NanoML production: the branch
    # holds +-boundaryEta for every SimTrackster (checked on the fine-calo tau and the
    # single-photon files), so the dataset, compute_norm_par.py and the evaluation DERIVE
    # the entry point's eta and phi from boundary_{x,y,z} and never use these two branches
    # (they are only loaded). `boundary_{px,py,pz}` is the particle's momentum vector (GeV)
    # at the crossing, i.e. the momentum of the entering particle: |p| equals
    # `regressed_energy` up to the particle mass (E^2 - p^2 = m^2, checked on the fine-calo
    # tau sample) and its direction differs from the position direction for secondaries
    # that do not point back to the origin (median angle ~4 deg, 1% above 60 deg). The
    # boundary position is the target of the entrance-position regression
    # (BOUNDARY_POSITION_FIELDS); the momentum is kept for a future direction target.
    # pdgId/eventId/bunchCrossing are not stored on the SimTrackster and are read off its
    # seed SimCluster, see PARTICLE_SEED_LOOKUP.
    "ticlSimTracksters": {
        "eta": "ticlSimTracksters_barycenter_eta",
        "phi": "ticlSimTracksters_barycenter_phi",
        "x": "ticlSimTracksters_barycenter_x",
        "y": "ticlSimTracksters_barycenter_y",
        "z": "ticlSimTracksters_barycenter_z",
        "recEnergy": "ticlSimTracksters_regressed_energy",
        "boundary_x": "ticlSimTracksters_boundaryX",
        "boundary_y": "ticlSimTracksters_boundaryY",
        "boundary_z": "ticlSimTracksters_boundaryZ",
        "boundary_eta": "ticlSimTracksters_boundaryEta",
        "boundary_phi": "ticlSimTracksters_boundaryPhi",
        "boundary_px": "ticlSimTracksters_boundaryPx",
        "boundary_py": "ticlSimTracksters_boundaryPy",
        "boundary_pz": "ticlSimTracksters_boundaryPz",
    },
}

# Canonical particle fields describing the seed particle at the HGCAL boundary: its entry
# point into the calorimeter and its momentum there. Optional: a collection provides them
# only if they are in its PARTICLE_FIELD_MAP entry, and a file may lack the branches (older
# productions); the dataset and the plots then skip them unless a config asks for a boundary
# target. The regression targets built from the position are `boundary_x`, `boundary_y`,
# `boundary_z`, `boundary_eta`, `boundary_sinphi`, `boundary_cosphi` (see
# dataset.dataset_maskformers.EXTRA_TARGETS and the eval VARIABLES). `boundary_eta` and
# `boundary_phi` of the targets are always derived from boundary_{x,y,z}, see boundary_eta_phi.
BOUNDARY_POSITION_FIELDS = ("boundary_x", "boundary_y", "boundary_z", "boundary_eta", "boundary_phi")
BOUNDARY_MOMENTUM_FIELDS = ("boundary_px", "boundary_py", "boundary_pz")
BOUNDARY_FIELDS = BOUNDARY_POSITION_FIELDS + BOUNDARY_MOMENTUM_FIELDS


def boundary_fields(particle_collection):
    """The boundary fields (canonical names, see BOUNDARY_FIELDS) that `particle_collection`
    provides, in BOUNDARY_FIELDS order."""
    fields = PARTICLE_FIELD_MAP[particle_collection]
    return [f for f in BOUNDARY_FIELDS if f in fields]


def has_boundary_position(particle_collection):
    """Whether `particle_collection` provides the entry point boundary_{x,y,z}."""
    fields = PARTICLE_FIELD_MAP.get(particle_collection, {})
    return all(f in fields for f in ("boundary_x", "boundary_y", "boundary_z"))


def boundary_eta_phi(x, y, z):
    """eta and phi of the entry point from its cartesian coordinates (numpy, awkward or torch
    arrays). This is what every consumer uses for the boundary eta/phi: the dumped
    boundaryEta is identical to it, the dumped boundaryPhi is broken (see PARTICLE_FIELD_MAP)."""
    import numpy as np
    rho = np.sqrt(x ** 2 + y ** 2)
    return np.arcsinh(z / rho), np.arctan2(y, x)

# Canonical particle fields that a collection does not carry itself but that can be read
# off the object that seeded it. `seed_idx` is the per-particle index into `seed_collection`
# (another top-level collection of the same parquet file) and `fields` maps canonical names
# to branches of that collection. ticlSimTracksters_seedIndex is documented as "the
# SimCluster table row (SimCluster seed) or the CaloParticle table row (CaloParticle
# seed)": TICL's SimTrackstersProducer seeds the SimTrackster with the CaloParticle itself
# when the CaloParticle crossed the HGCAL boundary intact (an unconverted photon, a muon,
# ...) and with the SimCluster otherwise (e.g. every decay product of a tau). The files
# carry no CaloParticle table, so a CaloParticle-seeded particle is resolved at load time
# to the most energetic SimCluster of its CaloParticle through `caloparticle_seed` (see
# dataset.caloparticle_seeds); SimCluster-seeded particles are looked up directly.
PARTICLE_SEED_LOOKUP = {
    "ticlSimTracksters": {
        "seed_collection": "SimCluster",
        "seed_idx": "ticlSimTracksters_seedIndex",
        "seed_is_caloparticle": "ticlSimTracksters_seedIsCaloParticle",
        "fields": {
            "pdgId": "SimCluster_pdgId",
            "eventId": "SimCluster_eventId",
            "bunchCrossing": "SimCluster_bunchCrossing",
        },
        # How to find the seeds of a CaloParticle, for CaloParticle-seeded particles: the
        # seeds' CaloParticle row, and the energy that picks one when there are several.
        "caloparticle_seed": {
            "group_idx": "SimCluster_CaloPartIdx",
            "energy": "SimCluster_boundaryEnergy",
        },
    },
}


def particle_field_map(particle_collection):
    return PARTICLE_FIELD_MAP[particle_collection]


# Raw branch holding the per-topocluster best-match index into the particle collection,
# keyed by (topocluster_collection, particle_collection). This is the one field whose
# name doesn't follow a uniform naming convention across collections.
MATCH_IDX_MAP = {
    ("RecHitHGC", "MergedSimCluster"): "RecHitHGC_MergedSimClusterBestMatchIdx",
    ("RecHitHGC", "SimCluster"): "RecHitHGC_SimClusterBestMatchIdx",
    ("LayerCluster", "SimCluster"): "LayerCluster_SimCluster_MatchIdx",
}

# (topocluster_collection, particle_collection) combinations whose raw match-index branch
# is a flattened many-to-many list of match candidates (a topocluster can match zero or
# several particles), together with the companion branches needed to reduce it down to a
# single best-match index per topocluster, and the direction (`best`) in which the quality
# branch is read. Combinations not listed here already provide a one-value-per-topocluster
# best-match index and need no reduction.
MANY_TO_MANY_MATCH_MAP = {
    ("LayerCluster", "SimCluster"): {
        "match_idx": "LayerCluster_SimCluster_MatchIdx",
        "match_qual": "LayerCluster_SimCluster_MatchQual",
        "num_match": "LayerCluster_SimClusterNumMatch",
        # Which candidate wins: "min" = smallest match_qual, "max" = largest. MatchQual is the
        # CMS layer-cluster -> SimCluster association SCORE (0 = the cluster's energy is fully
        # explained by that SimCluster), so the best candidate has the SMALLEST value. Checked
        # on taus_MergedCaloTruth/taus.parquet against the rechit-level
        # RecHitHGC_SimClusterBestMatchIdx energy majority: argmin agrees for 94% of the
        # multi-match clusters, argmax for 2%; single-match clusters carry MatchQual = 0 and
        # the per-cluster sum exceeds 1, so it is not a shared-energy fraction.
        "best": "min",
    },
}

# (topocluster_collection, particle_collection) combinations whose truth association is a
# TICL association table: a separate top-level collection of the parquet file, produced in
# CMSSW by AllTracksterToSimTracksterAssociatorsBy{Hits,LCs}Producer
# (SimCalorimetry/HGCalAssociatorProducers) and flattened by the NanoAOD
# AssociationOneToManyFlatTableProducer. It lists, for every topocluster, the particles it
# shares energy with, each with the shared energy (GeV) and an association score (lower is
# better). Since one topocluster can genuinely be shared between several particles, the
# incidence matrix for these pairs is built directly from the links, weighted by the
# fraction of the topocluster's energy that belongs to each particle (see
# HGCDataset._build_soft_links), instead of from a single best-match index. The
# per-topocluster best match (largest share) is still derived for the noise flag.
# `default` is the table used unless the dataset is given an explicit
# `association_collection`; `choices` are the tables known to describe the pair
# (...ByHits shares energy at the rechit level, ...ByLCs at the layer-cluster level).
ASSOCIATION_MAP = {
    ("ticlTrackstersCLUE3DHigh", "ticlSimTracksters"): {
        "default": "CLUE3DHighToSimTSByHits",
        "choices": ("CLUE3DHighToSimTSByHits", "CLUE3DHighToSimTSByLCs"),
    },
}


def association_fields(table):
    """Branch names of a NanoAOD one-to-many association table called `table`.

    Per topocluster: `n_links`, the number of linked particles. Per link, flattened within
    the event in topocluster order: `index` (index into the particle collection),
    `shared_energy` (GeV) and `score`. The table also carries per-topocluster offsets and a
    per-event total, which are redundant with `n_links` and not used.
    """
    return {
        "n_links": f"{table}_n{table}Links",
        "index": f"{table}Links_index",
        "shared_energy": f"{table}Links_sharedEnergy",
        "score": f"{table}Links_score",
    }


def supported_pairs():
    """All (topocluster_collection, particle_collection) combinations the dataset can load."""
    return sorted(set(MATCH_IDX_MAP) | set(ASSOCIATION_MAP))
