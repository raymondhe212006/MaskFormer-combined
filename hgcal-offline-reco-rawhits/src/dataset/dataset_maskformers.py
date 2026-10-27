import gc
import json
from pathlib import Path

import lightning as L
import numpy as np
import awkward as ak
import pyarrow.parquet as pq
import torch
import uproot
from lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1

from hepattn.utils.scaling import FeatureScaler

from utils.parquet import is_valid_file, list_parquet_files, read_parquet_files, split_parquet_files  # noqa: F401

# Particle classes (see HGCDataset.init_label_dicts): hadronic 0, electromagnetic 1, muon 2.
# The residual/fake class must equal hepattn's null class, which ObjectClassificationTask
# puts at index num_classes: configs need num_classes: 3 and a classifier net with 4 outputs.
NUM_CLASSES = 3
RESIDUAL_CLASS = NUM_CLASSES

# Regression targets (the `targets.particle` list of a config). BASE_TARGETS are always
# available: the particle energy and its barycenter / impact point. EXTRA_TARGETS are the
# entrance-position targets, built from the boundary fields of the particle collection
# (dataset.collections.BOUNDARY_POSITION_FIELDS: the point where the seed particle crossed the
# HGCAL front face), and map each target to the canonical topocluster field that serves as
# its NODE-LEVEL PROXY: hepattn's IncidenceBasedRegressionTask reads inputs["node_<field>"]
# for every regressed field and aggregates it over the predicted incidence matrix into a
# proxy particle (get_proxy_feats), so `node_boundary_x` is the raw node x (the trackster
# barycenter x for the trackster pair), `node_boundary_eta` the raw node eta, etc. The
# proxies are emitted only for the targets a config requests, see HGCDataset.__getitem__.
BASE_TARGETS = ("e", "eta", "sinphi", "cosphi", "r")
EXTRA_TARGETS = {
    "boundary_x": "x",
    "boundary_y": "y",
    "boundary_z": "z",
    "boundary_eta": "eta",
    "boundary_sinphi": "sinphi",
    "boundary_cosphi": "cosphi",
}
# Key of the node_raw_features dict of HGCDataset.load_event holding the raw value of each
# proxy field.
_PROXY_RAW_KEY = {"x": "raw_x", "y": "raw_y", "z": "raw_z", "eta": "raw_eta", "sinphi": "sinphi", "cosphi": "cosphi"}


def target_source(target):
    """Canonical particle field a regression target is built from (`particle_<field>` in
    the dataset): the energy for `e`, the 3D radius for `r`, the phi for sin/cos phi."""
    if target == "e":
        return "recEnergy"
    if target == "r":
        return "R"
    if target in ("sinphi", "cosphi"):
        return "phi"
    if target in ("boundary_sinphi", "boundary_cosphi"):
        return "boundary_phi"
    return target


def sparse_collate_fn(batch):
    """Custom collate that converts sparse incidence matrices to dense during batching."""
    inputs_list, labels_list = zip(*batch)
    
    inputs = {k: torch.stack([inp[k] for inp in inputs_list]) for k in inputs_list[0]}
    
    labels = {}
    for k in labels_list[0]:
        if k == "particle_incidence":
            
            labels[k] = [lab[k] for lab in labels_list]
        else:
            labels[k] = torch.stack([lab[k] for lab in labels_list])
    
    return inputs, labels

def normalize_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def do_padding(tensor, max_len):
    x = torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device)
    x[: len(tensor)] = tensor
    return x


def make_feature_scaler(scale_dict_path, strict):
    """FeatureScaler with `strict` where hepattn supports it.

    `strict=True` turns a variable missing from the scale dict into a KeyError instead of
    a silent identity transform. Neither upstream hepattn (samvanstroud/hepattn) nor the
    fork pinned in pyproject.toml (mmarchegiani/hepattn@python_3_12_3) accepts that
    argument (checked 2026-09-09), so the plain constructor would be the only one that
    works there; the `strict` call is kept for a hepattn that does support it. The
    fallback is the pre-`strict` behaviour (identity for missing keys) - see
    docs/NORMALIZATION_FIX.md section 2.5.
    """
    try:
        return FeatureScaler(scale_dict_path, strict=strict)
    except TypeError:
        if strict:
            print(f"WARNING: this hepattn has no FeatureScaler(strict=...); loading {scale_dict_path} "
                  "without strict checking (missing variables fall back to identity).")
        return FeatureScaler(scale_dict_path)


# Collection knowledge (raw branch names per collection, truth-association tables, supported
# pairs) lives in the torch-free module dataset/collections.py so that the evaluation plots can
# share it; the names are re-exported here so existing imports keep working.
from dataset.collections import (  # noqa: F401
    ASSOCIATION_MAP,
    BOUNDARY_FIELDS,
    BOUNDARY_MOMENTUM_FIELDS,
    BOUNDARY_POSITION_FIELDS,
    MANY_TO_MANY_MATCH_MAP,
    MATCH_IDX_MAP,
    OPTIONAL_TOPO_INPUTS,
    PARTICLE_FIELD_MAP,
    PARTICLE_SEED_LOOKUP,
    SIMCLUSTER_SUFFIX_TEMPLATES,
    TOPO_FIELD_MAP,
    association_fields,
    boundary_eta_phi,
    boundary_fields,
    has_boundary_position,
    particle_field_map,
    supported_pairs,
)
from dataset.caloparticle_seeds import resolve_caloparticle_seeds
from dataset.finite_checks import (
    INCIDENCE_TOL,
    assert_finite,
    assert_incidence_matrix,
    assert_scaler_finite,
    assert_transform_domain,
)


def reduce_to_best_match(topocluster, num_match_field, match_idx_field, match_qual_field, best="min"):
    """
    Reduce a flattened many-to-many match-candidate list (multiple match candidates per
    topocluster, concatenated within each event) to a single best-match index per
    topocluster, keeping the candidate with the smallest (`best="min"`) or largest
    (`best="max"`) match quality; the direction is a property of the quality branch and is
    recorded per pair in MANY_TO_MANY_MATCH_MAP (LayerCluster_SimCluster_MatchQual is an
    association score, lower is better). Topoclusters with no candidates (num_match == 0)
    get -1, matching the "no match" convention used by the one-per-topocluster BestMatchIdx
    branches.
    """
    if best not in ("min", "max"):
        raise ValueError(f"best must be 'min' or 'max', got {best!r}")
    num_match = topocluster[num_match_field]
    match_idx = topocluster[match_idx_field]
    match_qual = topocluster[match_qual_field]

    # Group match candidates by topocluster, ignoring event boundaries.
    flat_num_match = ak.flatten(num_match)
    per_topo_idx = ak.unflatten(ak.flatten(match_idx), flat_num_match)
    per_topo_qual = ak.unflatten(ak.flatten(match_qual), flat_num_match)

    # Re-impose event boundaries using the number of topoclusters per event.
    n_topo_per_event = ak.num(num_match, axis=1)
    match_idx_nested = ak.unflatten(per_topo_idx, n_topo_per_event)
    match_qual_nested = ak.unflatten(per_topo_qual, n_topo_per_event)

    argbest = ak.argmin if best == "min" else ak.argmax
    ibest = argbest(match_qual_nested, axis=-1, keepdims=True)
    return ak.fill_none(ak.firsts(match_idx_nested[ibest], axis=-1), -1)


class HGCDataset(Dataset):
    def __init__(
        self,
        filepath: str | list[str],
        inputs: dict,
        targets: dict,
        scale_dict_path: str,
        particle_scale_dict_path: str,
        topocluster_collection: str = "RecHitHGC",
        particle_collection: str = "MergedSimCluster",
        num_events: int = -1,
        num_objects: int = 150,
        max_nodes: int = 50000,
        remove_wrong_idxs: bool = True,
        incidence_cutval: float = 1e-4,
        normalize_incidence: bool = True,
        association_collection: str | None = None,
        min_link_fraction: float = 0.0,
        require_nodes_for_valid: bool = False,
        require_particles: bool = False,
        is_inference: bool = False,
        dummy_data: bool = False,
        verbose: bool = False,
        strict_scaling: bool = True
    ):
        super().__init__()

        self.sampling_seed = 42
        np.random.default_rng(self.sampling_seed)
        seed_everything(self.sampling_seed, workers=True)

        # strict_scaling=True turns a variable missing from the scale dict into a KeyError
        # instead of a silent identity transform. Set it to False only if you knowingly
        # want the fallback.
        #
        # Two scale dicts. `scale_dict_path` is fitted on the topocluster collection and
        # scales the node INPUT features; `particle_scale_dict_path` is fitted on the
        # particle collection and scales the regression TARGETS. Both use the keys
        # e/eta/r, but for populations whose spreads differ by orders of magnitude (a
        # rechit carries ~0.01 GeV, a SimCluster ~10 GeV), so they must be separate files.
        # The regression task (hepattn IncidenceBasedRegressionTask) and the prediction
        # writer must be given the PARTICLE dict: the task scales its incidence-aggregated
        # proxy features with it and multiplies its output onto them (mode: scale), and the
        # writer inverse-transforms the predictions with it.
        self.scaler = make_feature_scaler(scale_dict_path, strict_scaling)
        self.particle_scaler = make_feature_scaler(particle_scale_dict_path, strict_scaling)
        # A NaN/inf shift or a zero scale in a dict would turn every feature it scales into
        # NaN/inf: fail at construction instead.
        assert_scaler_finite(self.scaler, scale_dict_path)
        assert_scaler_finite(self.particle_scaler, particle_scale_dict_path)

        # Load parquet file with awkward array
        self.topocluster_collection = topocluster_collection
        self.particle_collection = particle_collection
        # Whether the topocluster collection carries a time branch (RecHitHGC, tracksters).
        # Decides whether `time` may be an input and whether the raw time is exported.
        self.has_time = "time" in TOPO_FIELD_MAP.get(topocluster_collection, {})
        # Optional node inputs (OPTIONAL_TOPO_INPUTS) requested by this config, as
        # {input name: canonical topocluster field}. Their branches are read only when
        # requested; the others are skipped when the raw branches are renamed below.
        self.optional_inputs = {
            name: canonical for name, canonical in OPTIONAL_TOPO_INPUTS.items() if name in inputs.get("topocluster", [])
        }
        # Soft (many-to-many, energy-weighted) truth association from a TICL association
        # table, see ASSOCIATION_MAP. `association_collection` selects the table;
        # `min_link_fraction` drops links carrying less than that fraction of the
        # topocluster's energy (0 keeps them all).
        self.has_soft_links = (topocluster_collection, particle_collection) in ASSOCIATION_MAP
        self.association_collection = association_collection
        self.min_link_fraction = min_link_fraction
        # Particles without any linked node are not valid targets when True: the class / mask heads are then
        # trained only on objects with evidence, and the Hungarian matching of the final layer (regression cost)
        # agrees with the mask-based matching of the intermediate layers (issue #5).
        self.require_nodes_for_valid = require_nodes_for_valid
        # Drop events without any particle: their targets are all invalid and the mask/incidence/
        # regression losses are 0/0 = NaN. Never happens on the toy (every SimCluster has a hit),
        # but does on the tau sets.
        self.require_particles = require_particles

        # Regression targets: every requested target must be one the dataset can build, and
        # the boundary (entrance-position) targets need the boundary fields of the particle
        # collection (see EXTRA_TARGETS / collections.boundary_fields).
        requested_targets = list(targets.get("particle", []))
        unknown_targets = [t for t in requested_targets if t not in BASE_TARGETS and t not in EXTRA_TARGETS]
        if unknown_targets:
            raise ValueError(
                f"unknown regression targets {unknown_targets} in targets.particle; "
                f"known: {list(BASE_TARGETS) + list(EXTRA_TARGETS)}"
            )
        self.extra_targets = [t for t in requested_targets if t in EXTRA_TARGETS]
        # Boundary fields (canonical names) this collection provides; the ones a file lacks
        # are dropped again when the file is read, unless a target needs them.
        self.boundary_fields = boundary_fields(particle_collection) if particle_collection in PARTICLE_FIELD_MAP else []
        missing_boundary = sorted({target_source(t) for t in self.extra_targets} - set(self.boundary_fields))
        if missing_boundary:
            raise ValueError(
                f"targets {self.extra_targets} need the boundary fields {missing_boundary} of the particle "
                f"collection, which PARTICLE_FIELD_MAP does not provide for '{particle_collection}' "
                f"(collections with a boundary position: {sorted(c for c in PARTICLE_FIELD_MAP if boundary_fields(c))})"
            )
        # Node-level proxies hepattn's regression task indexes for the extra targets,
        # {target: canonical topocluster field}, see EXTRA_TARGETS.
        self.proxy_targets = {t: EXTRA_TARGETS[t] for t in self.extra_targets}

        # init class dict
        self.init_label_dicts()

        # init variable lists
        self.init_variables_list()

        # input stuff: `filepath` is a parquet file, a directory of parquet files or a list of
        # files, read as one table in order. self.filepath is the display name.
        self.filepath = filepath if isinstance(filepath, (str, Path)) else f"{len(filepath)} parquet files"
        self.files = []
        self.dummy_data = dummy_data

        self.verbose = verbose

        # other properties
        self.inputs = inputs
        self.targets = targets
        self.num_objects = num_objects
        self.max_nodes = max_nodes
        #self.remove_wrong_idxs = remove_wrong_idxs
        self.incidence_cutval = incidence_cutval
        self.normalize_incidence = normalize_incidence
        self.is_inference = is_inference
        if dummy_data:
            print(f"Creating HGCAL dataset with dummy data and {num_events} samples")
            self.num_events = num_events if num_events > 0 else 1000
            return

        # Footers only: per-file event counts and schemas, without reading any data
        candidate_files = list_parquet_files(filepath)
        file_metadata = [pq.read_metadata(f) for f in candidate_files]
        # With num_events > 0, open only the leading files that hold that many events
        n_rows = np.cumsum([m.num_rows for m in file_metadata])
        n_used = len(candidate_files)
        if num_events > 0:
            n_used = min(n_used, int(np.searchsorted(n_rows, num_events)) + 1)
        self.files = candidate_files[:n_used]
        file_metadata = file_metadata[:n_used]
        print(f"Loading HGCAL dataset from {self.filepath} with {num_events} samples "
              f"({len(self.files)}/{len(candidate_files)} files, {int(n_rows[n_used - 1]):,} events in them)")
        if self.verbose:
            print(f"Is inference: {self.is_inference}")

        if topocluster_collection not in TOPO_FIELD_MAP:
            raise ValueError(
                f"Unsupported topocluster_collection '{topocluster_collection}'. "
                f"Supported: {sorted(TOPO_FIELD_MAP)}"
            )
        if "time" in inputs.get("topocluster", []) and not self.has_time:
            raise ValueError(
                f"'time' input is only supported for topocluster collections with a time branch "
                f"({sorted(c for c, f in TOPO_FIELD_MAP.items() if 'time' in f)}), got '{topocluster_collection}'"
            )
        for name, canonical in self.optional_inputs.items():
            if canonical not in TOPO_FIELD_MAP[topocluster_collection]:
                raise ValueError(
                    f"'{name}' input is only supported for topocluster collections with a '{canonical}' field "
                    f"({sorted(c for c, f in TOPO_FIELD_MAP.items() if canonical in f)}), got '{topocluster_collection}'"
                )
        pair = (topocluster_collection, particle_collection)
        if pair not in supported_pairs():
            raise ValueError(
                f"Unsupported (topocluster_collection, particle_collection) combination "
                f"('{topocluster_collection}', '{particle_collection}'). "
                f"Supported combinations: {supported_pairs()}"
            )
        if self.has_soft_links:
            choices = ASSOCIATION_MAP[pair]["choices"]
            if self.association_collection is None:
                self.association_collection = ASSOCIATION_MAP[pair]["default"]
            elif self.association_collection not in choices:
                raise ValueError(
                    f"association_collection '{self.association_collection}' does not describe the pair {pair}. "
                    f"Known tables: {list(choices)}"
                )
        elif self.association_collection is not None:
            raise ValueError(
                f"association_collection is only used by the pairs {sorted(ASSOCIATION_MAP)}, which read a TICL "
                f"association table; {pair} uses a per-topocluster match index instead"
            )

        print(f"Using topocluster collection '{topocluster_collection}' and particle collection '{particle_collection}'")
        if self.has_soft_links:
            print(f"Truth association from TICL association table '{self.association_collection}' "
                  f"(soft incidence, min_link_fraction={self.min_link_fraction:g})")

        topo_raw_fields = TOPO_FIELD_MAP[topocluster_collection]
        particle_raw_fields = particle_field_map(particle_collection)

        # Read only the top-level collections this pair needs. The files carry many more
        # (all rechits, every TICL iteration, every association table): the fine-calo tau
        # file is 2 GB of which the trackster pair needs a few MB.
        needed = [topocluster_collection, particle_collection]
        if particle_collection in PARTICLE_SEED_LOOKUP:
            needed.append(PARTICLE_SEED_LOOKUP[particle_collection]["seed_collection"])
        if self.has_soft_links:
            needed.append(self.association_collection)
        for f, meta in zip(self.files, file_metadata):
            available = set(meta.schema.to_arrow_schema().names)
            missing = [c for c in needed if c not in available]
            if missing:
                raise ValueError(f"{f} lacks the collections {missing} needed for {pair} (has: {sorted(available)})")
        self.df = ak.from_arrow(read_parquet_files(self.files, columns=needed))
        if num_events > 0:
            self.df = self.df[:num_events]
        self.topocluster = self.df[topocluster_collection]
        self.particle = self.df[particle_collection]

        # Rename raw branches to canonical field names ("topo_*" / "particle_*"), so
        # everything downstream of this point is agnostic to the chosen collections.
        skipped = set(OPTIONAL_TOPO_INPUTS.values()) - set(self.optional_inputs.values())
        for canonical, raw in topo_raw_fields.items():
            if canonical in skipped:
                continue
            if raw not in self.topocluster.fields:
                raise ValueError(f"'{topocluster_collection}' in {self.filepath} has no branch '{raw}' (canonical field '{canonical}')")
            self.topocluster[f"topo_{canonical}"] = self.topocluster[raw]

        # The boundary fields are optional in the file (older productions lack the branches):
        # skip a missing one unless a target needs it.
        needed_boundary = {target_source(t) for t in self.extra_targets}
        missing_boundary = []
        for canonical, raw in particle_raw_fields.items():
            if raw not in self.particle.fields:
                if canonical in BOUNDARY_FIELDS and canonical not in needed_boundary:
                    missing_boundary.append(canonical)
                    continue
                raise ValueError(f"'{particle_collection}' in {self.filepath} has no branch '{raw}' (canonical field '{canonical}')")
            self.particle[f"particle_{canonical}"] = self.particle[raw]
        if missing_boundary:
            print(f"NOTE: '{particle_collection}' in {self.filepath} has no {[particle_raw_fields[c] for c in missing_boundary]} "
                  f"branch(es): the boundary fields {missing_boundary} are not loaded")
            self.particle_variables = [v for v in self.particle_variables if v[len("particle_"):] not in missing_boundary]
        if particle_collection in PARTICLE_SEED_LOOKUP:
            self._lookup_seed_fields(PARTICLE_SEED_LOOKUP[particle_collection])
        for canonical in ("pdgId", "eventId", "bunchCrossing"):
            if f"particle_{canonical}" not in self.particle.fields:
                raise ValueError(
                    f"particle collection '{particle_collection}' provides no '{canonical}': add it to "
                    f"PARTICLE_FIELD_MAP or PARTICLE_SEED_LOOKUP"
                )

        # Truth association, reduced to canonical fields. Every pair gets a per-topocluster
        # best-match index `topo_match_idx` (-1 = no match), used for the noise/pileup flags
        # and, for the single-index pairs, for the incidence matrix. Soft pairs additionally
        # keep every (topocluster, particle) link with its energy share for the incidence.
        if self.has_soft_links:
            self._build_soft_links(particle_raw_fields["recEnergy"])
        elif pair in MANY_TO_MANY_MATCH_MAP:
            fields = MANY_TO_MANY_MATCH_MAP[pair]
            self.topocluster["topo_match_idx"] = reduce_to_best_match(
                self.topocluster, fields["num_match"], fields["match_idx"], fields["match_qual"],
                best=fields.get("best", "min"),
            )
        else:
            self.topocluster["topo_match_idx"] = self.topocluster[MATCH_IDX_MAP[pair]]

        # Derive kinematics not provided directly by the raw collection
        topo_x = self.topocluster["topo_x"]
        topo_y = self.topocluster["topo_y"]
        topo_z = self.topocluster["topo_z"]
        if "topo_eta" not in self.topocluster.fields:
            T = np.sqrt(topo_x**2 + topo_y**2)  # distance in transverse plane
            theta = np.arctan2(T, topo_z)
            self.topocluster["topo_eta"] = -np.log(np.tan(theta / 2))
        if "topo_phi" not in self.topocluster.fields:
            self.topocluster["topo_phi"] = np.arctan2(topo_y, topo_x)  # phi=0 points vertically upward
        topo_phi = self.topocluster["topo_phi"]
        self.topocluster["topo_R"] = np.sqrt(topo_x**2 + topo_y**2 + topo_z**2)
        self.topocluster["topo_cosphi"] = np.cos(topo_phi)
        self.topocluster["topo_sinphi"] = np.sin(topo_phi)

        # Noise/pileup flags, derived generically from the match index and the
        # matched particle's eventId/bunchCrossing
        match_idx = self.topocluster["topo_match_idx"]  # -1 for noise
        safe_idx = ak.where(match_idx >= 0, match_idx, 0)  # avoid out-of-bounds indexing

        def lookup(field):
            # Pad every event to at least one particle so that index 0 is valid in events
            # that have topoclusters (all noise) but no particle at all, e.g. the empty
            # endcap of an endcap-split event; the dummy value never enters the flags
            # because they require match_idx >= 0.
            padded = ak.fill_none(ak.pad_none(self.particle[field], 1, axis=1), 0)
            return padded[safe_idx]

        particle_event_id = lookup("particle_eventId")
        particle_bunch_crossing = lookup("particle_bunchCrossing")
        self.topocluster["topo_is_noise"] = match_idx == -1
        self.topocluster["topo_is_pileup"] = ((particle_event_id > 0) | (particle_bunch_crossing > 0)) & (match_idx >= 0)

        # Add radius branch to particle collection
        particle_x = self.particle["particle_x"]
        particle_y = self.particle["particle_y"]
        particle_z = self.particle["particle_z"]
        self.particle["particle_R"] = np.sqrt(particle_x**2 + particle_y**2 + particle_z**2)

        # The entry point's eta/phi are derived from its position and overwrite whatever the
        # boundaryEta/boundaryPhi branches hold: the dumped phi is +-eta in the
        # CMSSW_20_0_0_pre1 production (dataset.collections.boundary_eta_phi).
        if has_boundary_position(particle_collection) and all(
            f"particle_boundary_{c}" in self.particle.fields for c in ("x", "y", "z")
        ):
            b_eta, b_phi = boundary_eta_phi(
                self.particle["particle_boundary_x"], self.particle["particle_boundary_y"], self.particle["particle_boundary_z"]
            )
            self.particle["particle_boundary_eta"] = b_eta
            self.particle["particle_boundary_phi"] = b_phi
            for canonical in ("boundary_eta", "boundary_phi"):
                if f"particle_{canonical}" not in self.particle_variables:
                    self.particle_variables.append(f"particle_{canonical}")

        self.num_events = len(self.df) if num_events < 0 else min(len(self.df), num_events)
        self.load_data()

    def _lookup_seed_fields(self, lookup):
        """Fill canonical particle fields from the collection that seeded the particles.

        For ticlSimTracksters: pdgId/eventId/bunchCrossing of the seed SimCluster, indexed by
        ticlSimTracksters_seedIndex. Verified on the fine-calo tau sample: the SimTrackster's
        boundary position equals SimCluster_impactPoint_* at that index to <0.01 cm and its
        regressed_energy equals SimCluster_boundaryEnergy, so the two tables are aligned.

        Particles seeded by the CaloParticle itself (it crossed the HGCAL boundary intact, e.g.
        an unconverted photon; never the case for a tau's decay products) carry a CaloParticle
        row in seedIndex instead. They are mapped to the most energetic SimCluster of that
        CaloParticle first, see dataset.caloparticle_seeds.
        """
        seed_collection = lookup["seed_collection"]
        if seed_collection not in self.df.fields:
            raise ValueError(
                f"particle collection '{self.particle_collection}' takes {sorted(lookup['fields'])} from its seed "
                f"collection '{seed_collection}', which is not in {self.filepath} (has: {list(self.df.fields)})"
            )
        seeds = self.df[seed_collection]
        seed_idx = self.particle[lookup["seed_idx"]]

        flag = lookup.get("seed_is_caloparticle")
        is_cp = self.particle[flag] if flag is not None and flag in self.particle.fields else None
        n_cp = int(ak.sum(is_cp)) if is_cp is not None else 0
        if n_cp:
            cp = lookup.get("caloparticle_seed")
            if cp is None or cp["group_idx"] not in seeds.fields:
                raise ValueError(
                    f"{n_cp} '{self.particle_collection}' particles are seeded by a CaloParticle, so their "
                    f"{lookup['seed_idx']} does not index '{seed_collection}', and '{seed_collection}' has no "
                    f"CaloParticle row branch ({cp['group_idx'] if cp else 'none configured'}) to resolve them"
                )
            energy = seeds[cp["energy"]] if cp.get("energy") in seeds.fields else None
            seed_idx, n_unresolved = resolve_caloparticle_seeds(seed_idx, is_cp, seeds[cp["group_idx"]], energy)
            if n_unresolved:
                raise ValueError(
                    f"{n_unresolved} CaloParticle-seeded '{self.particle_collection}' particles have no "
                    f"'{seed_collection}' with {cp['group_idx']} equal to their {lookup['seed_idx']}"
                )
            print(
                f"{n_cp} '{self.particle_collection}' particles are seeded by a CaloParticle (it entered HGCAL "
                f"intact); their {sorted(lookup['fields'])} are taken from the most energetic '{seed_collection}' "
                f"of that CaloParticle"
            )
        n_seeds = ak.num(seeds[next(iter(lookup["fields"].values()))], axis=1)
        if ak.any(seed_idx < 0) or ak.any(seed_idx >= n_seeds):
            raise ValueError(f"{lookup['seed_idx']} out of range of the '{seed_collection}' collection")

        for canonical, raw in lookup["fields"].items():
            self.particle[f"particle_{canonical}"] = seeds[raw][seed_idx]

    def _build_soft_links(self, particle_count_field):
        """Read the TICL association table of this pair (see ASSOCIATION_MAP).

        Sets on self.topocluster
          topo_match_idx      particle carrying the largest share of each topocluster's energy,
                              -1 for a topocluster without (surviving) links;
        and on self, as per-event jagged arrays with one entry per (topocluster, particle) link
          link_node_idx       index of the topocluster within the event,
          link_particle_idx   index of the particle within the event,
          link_weight         sharedEnergy / topocluster energy, i.e. the fraction of the
                              topocluster's energy that belongs to that particle.
        Over its links the weights of one topocluster add up to at most 1; the shortfall is
        energy of the topocluster without any simulated contribution (fine-calo tau sample:
        every trackster has at least one link, the sum is 1 for the median trackster and
        below 0.99 for 2.7% of them, never below 0.91). Links below `min_link_fraction` are
        dropped before the best match is chosen.
        """
        table = self.association_collection
        if table not in self.df.fields:
            raise ValueError(f"association table '{table}' is not in {self.filepath} (has: {list(self.df.fields)})")
        fields = association_fields(table)
        assoc = self.df[table]
        missing = [f for f in fields.values() if f not in assoc.fields]
        if missing:
            raise ValueError(f"association table '{table}' lacks the branches {missing} (has: {list(assoc.fields)})")

        topo_energy = self.topocluster["topo_energy"]
        n_topo = ak.num(topo_energy, axis=1)

        # The shares divide by the topocluster energy: a non-finite or non-positive energy
        # would give inf/NaN shares that only surface later as a NaN incidence loss.
        flat_energy = ak.to_numpy(ak.flatten(topo_energy))
        assert_finite(f"'{self.topocluster_collection}' energy", flat_energy, context=str(self.filepath))
        if (flat_energy <= 0).any():
            raise AssertionError(
                f"{int((flat_energy <= 0).sum())} '{self.topocluster_collection}' topoclusters have energy <= 0 "
                f"in {self.filepath}: their energy shares are undefined"
            )
        flat_shared = ak.to_numpy(ak.flatten(assoc[fields["shared_energy"]]))
        assert_finite(f"'{fields['shared_energy']}'", flat_shared, context=str(self.filepath))
        if (flat_shared < 0).any():
            raise AssertionError(f"{int((flat_shared < 0).sum())} '{fields['shared_energy']}' entries are negative in {self.filepath}")
        n_links = assoc[fields["n_links"]]
        if ak.any(ak.num(n_links, axis=1) != n_topo):
            raise ValueError(
                f"association table '{table}' has one row per topocluster but its multiplicity differs from "
                f"'{self.topocluster_collection}' in some events"
            )

        # Links are stored flattened within the event, in topocluster order: nest them per
        # topocluster (event -> topocluster -> links), then re-impose the event boundaries.
        flat_n_links = ak.flatten(n_links)

        def nest(per_event_flat):
            return ak.unflatten(ak.unflatten(ak.flatten(per_event_flat), flat_n_links), n_topo)

        link_idx = nest(assoc[fields["index"]])
        link_shared = nest(assoc[fields["shared_energy"]])

        n_particles = ak.num(self.particle[particle_count_field], axis=1)
        if ak.any(ak.flatten(ak.flatten(link_idx, axis=-1) >= n_particles, axis=None)) or ak.any(link_idx < 0):
            raise ValueError(f"'{fields['index']}' points outside the '{self.particle_collection}' collection")

        # Fraction of the topocluster's energy shared with each linked particle. The
        # (event, topocluster) energies broadcast over each topocluster's link list.
        weight = link_shared / topo_energy
        # Every share and the shares of one topocluster summed over its links must lie in
        # [0, 1] (up to float32 rounding, INCIDENCE_TOL): the incidence matrix is built from
        # them unchanged, and the incidence task's KL loss assumes that normalisation.
        flat_weight = ak.to_numpy(ak.flatten(weight, axis=None))
        assert_finite("energy share (sharedEnergy / topocluster energy)", flat_weight, context=str(self.filepath))
        share_sum = ak.to_numpy(ak.flatten(ak.sum(weight, axis=-1)))
        if flat_weight.max(initial=0.0) > 1 + INCIDENCE_TOL or share_sum.max(initial=0.0) > 1 + INCIDENCE_TOL:
            raise AssertionError(
                f"energy shares exceed 1 in {self.filepath} (max share {flat_weight.max(initial=0.0):.6f}, max sum over the "
                f"links of one topocluster {share_sum.max(initial=0.0):.6f}): the association table '{table}' does not "
                f"describe the energies of '{self.topocluster_collection}'"
            )
        if self.min_link_fraction > 0:
            keep = weight >= self.min_link_fraction
            link_idx, weight = link_idx[keep], weight[keep]

        best = ak.argmax(weight, axis=-1, keepdims=True)
        best_idx = ak.fill_none(ak.firsts(link_idx[best], axis=-1), -1)
        self.topocluster["topo_match_idx"] = ak.values_astype(best_idx, np.int64)

        node_idx = ak.broadcast_arrays(ak.local_index(topo_energy, axis=1), weight)[0]
        self.link_node_idx = ak.flatten(node_idx, axis=-1)
        self.link_particle_idx = ak.flatten(link_idx, axis=-1)
        self.link_weight = ak.flatten(weight, axis=-1)

    def load_data(self):
        self.full_data_array = {}

        for var in tqdm(self.topo_variables):
            self.full_data_array[var] = self.topocluster[var]
            if var == self.topo_variables[0]:  # just take the first variable to get the number of topoclusters
                self.n_topos = np.array([len(x) for x in self.full_data_array[var]])

        for var in tqdm(self.particle_variables):
            self.full_data_array[var] = self.particle[var]
            if not "particle_recEnergy" in self.particle.fields:
                raise ValueError("Missing required variable 'particle_recEnergy' in particle collection")
            elif var == "particle_recEnergy":
                self.n_particles = np.array([len(x) for x in self.full_data_array[var]])

        self.event_number = np.arange(len(self.n_topos))
        if self.has_soft_links:
            self.n_links = ak.to_numpy(ak.num(self.link_weight, axis=1))

        # Events without nodes carry no input (e.g. 22% of the tau rows have no CLUE3DHigh trackster,
        # including every row without SimTracksters): they give fully masked attention and empty
        # mask/incidence targets, so they are dropped.
        mask = (self.n_topos > 0) & (self.n_topos < self.max_nodes) & (self.n_particles < self.num_objects)
        if self.require_particles:
            # A particle is only a target if a node is matched to it (require_nodes_for_valid), so
            # an event needs a particle AND a matched node; otherwise every target is invalid.
            if self.has_soft_links:
                has_target = self.n_links > 0
            else:
                has_target = ak.to_numpy(ak.any(self.topocluster["topo_match_idx"] >= 0, axis=1))
            self.has_target = has_target & (self.n_particles > 0)
            mask &= self.has_target
        print(f"Removing {(~mask).sum()} events with no nodes or too many nodes or particles")
        print(f"{(self.n_topos == 0).sum()}/{self.num_events} events with no nodes")
        print(f"{(self.n_topos >= self.max_nodes).sum()}/{self.num_events} events with too many nodes")
        print(f"{(self.n_particles >= self.num_objects).sum()}/{self.num_events} events with too many particles")

        # Event counts and cutflow, written to dataset_counts.json in the run folder by
        # HGCDataModule. "failing" counts each selection on its own, "cutflow" applies them in order.
        selections = {
            "n_nodes > 0": self.n_topos > 0,
            f"n_nodes < max_nodes ({self.max_nodes})": self.n_topos < self.max_nodes,
            f"n_particles < num_objects ({self.num_objects})": self.n_particles < self.num_objects,
        }
        if self.require_particles:
            selections["has a particle with a matched node"] = self.has_target
        cutflow = {"loaded": int(len(self.n_topos))}
        passing = np.ones(len(self.n_topos), dtype=bool)
        for name, sel in selections.items():
            passing &= sel
            cutflow[name] = int(passing.sum())
        # Efficiency of each step: cumulative w.r.t. the loaded events, relative w.r.t. the previous step
        steps = list(cutflow.values())
        cutflow_efficiency = {
            name: {
                "cumulative": n / steps[0] if steps[0] else 0.0,
                "relative": n / prev if prev else 0.0,
            }
            for name, n, prev in zip(list(cutflow)[1:], steps[1:], steps[:-1])
        }

        def multiplicity(n):
            n = np.asarray(n)
            if len(n) == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}
            return {"mean": float(n.mean()), "std": float(n.std()), "min": int(n.min()), "max": int(n.max())}
        self.counts = {
            "filepath": str(self.filepath),
            "files": [str(f) for f in self.files],
            "topocluster_collection": self.topocluster_collection,
            "particle_collection": self.particle_collection,
            "events_in_files": int(sum(pq.read_metadata(f).num_rows for f in self.files)),
            "failing": {
                "no nodes": int((self.n_topos == 0).sum()),
                "no particles": int((self.n_particles == 0).sum()),
                "too many nodes": int((self.n_topos >= self.max_nodes).sum()),
                "too many particles": int((self.n_particles >= self.num_objects).sum()),
            },
            "cutflow": cutflow,
            "cutflow_efficiency": cutflow_efficiency,
            "selected": {
                "events": int(mask.sum()),
                "nodes": int(self.n_topos[mask].sum()),
                "particles": int(self.n_particles[mask].sum()),
            },
            # Per-event multiplicities, before ("loaded") and after ("selected") the selections
            "topocluster_counts": {"loaded": multiplicity(self.n_topos), "selected": multiplicity(self.n_topos[mask])},
            "particle_counts": {"loaded": multiplicity(self.n_particles), "selected": multiplicity(self.n_particles[mask])},
        }
        #if self.remove_wrong_idxs:
        #    track_particle_idxs = tree["track_particle_idx"].array(library="np", entry_stop=self.num_events)
        #    track_particle_idxs_lens = np.array([len(el) for el in track_particle_idxs])
        #    print(f"Removing {np.sum(track_particle_idxs_lens != self.n_tracks)} events with mismatching track_particle_idx")
        #    mask &= track_particle_idxs_lens == self.n_tracks

        self.n_topos = self.n_topos[mask]
        self.n_particles = self.n_particles[mask]
        self.event_number = self.event_number[mask]

        for var in self.topo_variables + self.particle_variables:
            # flatten the arrays
            self.full_data_array[var] = np.concatenate(self.full_data_array[var][mask])
            # get particle class labels
            if var == "particle_pdgId":
                self.particle_class = torch.tensor([self.class_labels[x] for x in self.full_data_array[var]])

        # transform variables and transform to tensors
        for key, val in self.full_data_array.items():
            if self.verbose:
                print(f"Converting variable `{key}` to torch.tensor")
            self.full_data_array[key] = torch.tensor(val.to_numpy()) # Convert awkward array to numpy first

        # Every raw branch this pair uses must be finite, and each scaled variable must lie in
        # the domain of its transform (fn: log needs > 0, log1p > -1): one bad entry would become
        # a NaN feature or target of that event and, through the loss, a NaN epoch mean. Checked
        # once here on the flat arrays; load_event re-checks the scaled values per event.
        for key, val in self.full_data_array.items():
            assert_finite(f"branch '{key}'", val, context=str(self.filepath))
        domain_checks = [
            ("topocluster energy (input 'e')", "topo_energy", self.scaler.transforms["e"]),
            ("topocluster eta (input 'eta')", "topo_eta", self.scaler.transforms["eta"]),
            ("topocluster R (input 'r')", "topo_R", self.scaler.transforms["r"]),
            ("particle energy (target 'e')", "particle_recEnergy", self.particle_scaler.transforms["e"]),
            ("particle eta (target 'eta')", "particle_eta", self.particle_scaler.transforms["eta"]),
            ("particle R (target 'r')", "particle_R", self.particle_scaler.transforms["r"]),
            *(
                (f"topocluster {canonical} (input '{name}')", f"topo_{canonical}", self.scaler.transforms[name])
                for name, canonical in self.optional_inputs.items()
            ),
        ]
        for name in self.extra_targets:
            source = target_source(name)
            if source == "boundary_phi":
                continue  # sin/cos phi: bounded, no transform to check
            domain_checks.append((f"particle {source} (target '{name}')", f"particle_{source}", self.particle_scaler.transforms[name]))
        for name, key, transform in domain_checks:
            assert_transform_domain(name, self.full_data_array[key], transform, context=str(self.filepath))

        # Extract and store match indices separately (removed from full_data_array to save memory)
        self.match_indices = self.full_data_array.pop("topo_match_idx")
        if self.verbose:
            print("Stored match indices as separate flat tensor")

        self.topo_cumsum = np.cumsum([0, *self.n_topos.tolist()])
        self.particle_cumsum = np.cumsum([0, *self.n_particles.tolist()])

        if self.has_soft_links:
            # Flatten the surviving events' links into flat arrays sliced per event via
            # link_cumsum, like the node/particle arrays above.
            self.n_links = self.n_links[mask]
            self.link_node_idx = ak.to_numpy(ak.flatten(self.link_node_idx[mask])).astype(np.int64)
            self.link_particle_idx = ak.to_numpy(ak.flatten(self.link_particle_idx[mask])).astype(np.int64)
            self.link_weight = ak.to_numpy(ak.flatten(self.link_weight[mask])).astype(np.float32)
            self.link_cumsum = np.cumsum([0, *self.n_links.tolist()])
            n_nodes_total = int(self.n_topos.sum())
            n_links_total = len(self.link_weight)
            below_cut = float(np.mean(self.link_weight <= self.incidence_cutval)) if n_links_total else 0.0
            print(
                f"Soft incidence from '{self.association_collection}': {n_links_total:,} links for {n_nodes_total:,} nodes "
                f"({n_links_total / max(n_nodes_total, 1):.2f} per node); {below_cut:.1%} of the links carry at most "
                f"incidence_cutval={self.incidence_cutval:g} of their node's energy and will not enter the mask target"
            )

        # needed for batch sampling
        self.n_nodes = self.n_topos

        self.aux_data_array = {}
        for var in self.aux_vars:
            print(f"Loading aux var {var}")
            self.aux_data_array[var] = arrays[var][mask]

        self.num_events = np.sum(mask)
        print(f"Number of events after filtering: {self.num_events}")

        # Free memory: delete large awkward arrays that are no longer needed
        del self.df, self.topocluster, self.particle
        gc.collect()
        if self.verbose:
            print("Freed memory: deleted df, topocluster, and particle awkward arrays")

        self.neg_contribs = []
        self.fake_TC_count = []

    def __len__(self):
        return int(self.num_events)

    def load_event(self, idx):
        """This function loads the data for a single event and processes it to return the node features, particle data, and incidence matrix.
        It also applies padding to ensure consistent tensor sizes across events.
        """
        n_topos = self.n_topos[idx]
        n_particles = self.n_particles[idx]

        n_nodes = n_topos

        topo_start, topo_end = self.topo_cumsum[idx], self.topo_cumsum[idx + 1]
        particle_start, particle_end = (
            self.particle_cumsum[idx],
            self.particle_cumsum[idx + 1],
        )

        # topo features
        topo_e = self.full_data_array["topo_energy"][topo_start:topo_end]
        if self.has_time:
            topo_time = self.full_data_array["topo_time"][topo_start:topo_end]
        topo_x = self.full_data_array["topo_x"][topo_start:topo_end]
        topo_y = self.full_data_array["topo_y"][topo_start:topo_end]
        topo_z = self.full_data_array["topo_z"][topo_start:topo_end]
        topo_eta = self.full_data_array["topo_eta"][topo_start:topo_end]
        topo_phi = self.full_data_array["topo_phi"][topo_start:topo_end]
        topo_R = self.full_data_array["topo_R"][topo_start:topo_end]
        topo_cosphi = self.full_data_array["topo_cosphi"][topo_start:topo_end]
        topo_sinphi = self.full_data_array["topo_sinphi"][topo_start:topo_end]
        topo_is_noise = self.full_data_array["topo_is_noise"][topo_start:topo_end]
        topo_is_pileup = self.full_data_array["topo_is_pileup"][topo_start:topo_end]
        #topo_energy_ecal = self.full_data_array["topo_energy_ecal"][topo_start:topo_end]
        #topo_energy_hcal = self.full_data_array["topo_energy_hcal"][topo_start:topo_end]
        #topo_energy_other = self.full_data_array["topo_energy_other"][topo_start:topo_end]
        #topo_em_frac = topo_energy_ecal / (topo_energy_ecal + topo_energy_hcal + topo_energy_other)

        # WARNING: These operations are executed for every event.
        # What we could do for speed up, instead, is to compute these operations on Awkward arrays in load_data()

        #T = np.sqrt(topo_x**2 + topo_y**2) # distance in Transverse plane
        #theta = np.arctan2(T, topo_z)
        #topo_eta = -np.log(np.tan(theta/2))
        #topo_R = np.sqrt(topo_x**2 + topo_y**2 + topo_z**2)
        #topo_phi = np.arctan2(topo_y, topo_x) # Assuming phi=0 points vertically upward
        #topo_cosphi = np.cos(topo_phi)
        #topo_sinphi = np.sin(topo_phi)

        node_features = {
            # common features
            #"pt": torch.cat(
            #    [
            #        #self.scaler.transforms["pt"].transform(track_pt),
            #        torch.zeros(n_topos, dtype=torch.float32),
            #    ],
            #    -1,
            #),
            "eta": torch.cat(
                [
                    #self.scaler.transforms["eta"].transform(track_eta),
                    self.scaler.transforms["eta"].transform(topo_eta),
                ],
                -1,
            ),
            "phi": torch.cat(
                [
                    #track_phi,
                    topo_phi,
                ],
                -1,
            ),
            "cosphi": torch.cat(
                [
                    #track_cosphi,
                    topo_cosphi
                ],
            -1
            ),
            "sinphi": torch.cat(
                [
                    #track_sinphi,
                    topo_sinphi
                ],
            -1
            ),
            "r": torch.cat(
                [
                    #torch.zeros(n_tracks, dtype=torch.float32),
                    # scaled like e/eta; raw R is ~320-570 cm and used to enter unscaled
                    self.scaler.transforms["r"].transform(topo_R),
                ],
                -1,
            ),
            "e": torch.cat(
                [
                    #torch.zeros(n_tracks, dtype=torch.float32),
                    self.scaler.transforms["e"].transform(topo_e),
                ],
                -1,
            ),
            "is_topo": torch.cat(
                [
                    #torch.zeros(n_tracks, dtype=torch.float32),
                    torch.ones(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
        }

        if self.has_time:
            node_features["time"] = torch.cat(
                [
                    self.scaler.transforms["t"].transform(topo_time),
                ],
                -1,
            )

        # Optional inputs are scaled under their own name, e.g. `e_em` (trackster
        # raw_em_energy, fn: log1p since it is 0 for tracksters without CE-E layer clusters)
        optional_raw = {
            name: self.full_data_array[f"topo_{canonical}"][topo_start:topo_end]
            for name, canonical in self.optional_inputs.items()
        }
        for name, raw in optional_raw.items():
            node_features[name] = self.scaler.transforms[name].transform(raw)

        node_raw_features = {
            "raw_e": torch.cat([
                #torch.zeros(n_tracks, dtype=torch.float32),
                topo_e
                ],
                -1
            ),
            "raw_pt": torch.cat(
                [
            #        track_pt,
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "raw_eta": torch.cat([topo_eta], -1),
            "raw_phi": torch.cat([topo_phi], -1),
            "sinphi": torch.cat(
                [
                    #track_sinphi,
                    topo_sinphi
                ],
                -1
            ),
            "cosphi": torch.cat(
                [
                    #track_cosphi,
                    topo_cosphi
                ],
                -1
            ),
            "r": torch.cat(
                [
                    #torch.zeros(n_tracks, dtype=torch.float32),
                    topo_R,
                ],
                -1,
            ),
            "is_track": torch.cat(
                [
                    #torch.ones(n_tracks, dtype=torch.float32),
                    torch.zeros(n_topos, dtype=torch.float32),
                ],
                -1,
            ),
            "is_noise": torch.cat(
                [
                    #track_is_noise,
                    topo_is_noise
                ],
            -1
            ),
            "is_pileup": torch.cat(
                [
                    #track_is_pileup,
                    topo_is_pileup
                ],
            -1
            ),
        }

        if self.has_time:
            node_raw_features["raw_time"] = torch.cat([topo_time], -1)
        for name, raw in optional_raw.items():
            node_raw_features[f"raw_{name}"] = raw
        # Raw node position for the proxies of the boundary targets (EXTRA_TARGETS): the
        # trackster/layer-cluster/rechit x, y, z; eta and sin/cos phi are already there.
        for canonical in set(self.proxy_targets.values()):
            key = _PROXY_RAW_KEY[canonical]
            if key not in node_raw_features:
                node_raw_features[key] = self.full_data_array[f"topo_{canonical}"][topo_start:topo_end]

        for key, val in node_features.items():
            assert_finite(f"node input feature '{key}'", val, context=f"event {self.event_number[idx]}")
            val = do_padding(val, self.max_nodes)
            node_features[key] = val

        for key, val in node_raw_features.items():
            val = do_padding(val, self.max_nodes)
            node_raw_features[key] = val

        particle_class = self.particle_class[particle_start:particle_end]

        #track_particle_idx = torch.tensor(self.aux_data_array["track_particle_idx"][idx], dtype=torch.int64)
        #if self.is_inference:
            #trackless_particle_mask = torch.zeros_like(particle_class, dtype=torch.bool)
        #else:
            #trackless_particle_mask = torch.ones_like(particle_class, dtype=torch.bool)
            #trackless_particle_mask[track_particle_idx] = False

            #trackless_chhad_and_e_mask = trackless_particle_mask & (particle_class < 2)
            #trackless_muon_mask = trackless_particle_mask & (particle_class == 2)

            # trackless ch hads and es become nu hads and photons (+3)
            #particle_class[trackless_chhad_and_e_mask] += 3

            # trackless muons become neutral hadrons
            #particle_class[trackless_muon_mask] = 3

        particle_data = {
            "e": self.full_data_array["particle_recEnergy"][particle_start:particle_end],
            "eta": self.full_data_array["particle_eta"][particle_start:particle_end],
            "sinphi": torch.sin(self.full_data_array["particle_phi"][particle_start:particle_end]),
            "cosphi": torch.cos(self.full_data_array["particle_phi"][particle_start:particle_end]),
            "r": self.full_data_array["particle_R"][particle_start:particle_end],
            "class": particle_class,
        }
        # Entrance-position targets (EXTRA_TARGETS): the boundary point of the seed particle,
        # scaled with their own entries of the particle dict (boundary_x, boundary_y, ...)
        for name in self.extra_targets:
            values = self.full_data_array[f"particle_{target_source(name)}"][particle_start:particle_end]
            if name.endswith("sinphi"):
                values = torch.sin(values)
            elif name.endswith("cosphi"):
                values = torch.cos(values)
            particle_data[name] = values
        # Particle-fitted transforms, NOT self.scaler (topocluster-fitted): the two share
        # the keys e/eta/r. The regression task scales its proxy features with the same
        # particle dict, so targets and proxies live in one space.
        particle_data = self.particle_scaler.transform(particle_data)
        for key, val in particle_data.items():
            assert_finite(f"particle target '{key}'", val, context=f"event {self.event_number[idx]}")
            val = do_padding(val, self.num_objects)
            particle_data[key] = val

        #has_track = torch.zeros(self.num_objects, dtype=torch.bool)
        #has_track[:n_particles] = ~trackless_particle_mask

        node_q_mask = torch.zeros(self.max_nodes, dtype=torch.bool)
        node_q_mask[:n_nodes] = True

        indicator = torch.zeros(self.num_objects)

        #part_idx = self.aux_data_array["track_particle_idx"][idx]
        #track_idx = np.arange(len(part_idx))
        #if self.is_inference:
        #    part_idx[part_idx < 0] = 0
        # print(incidence_matrix.shape, part_idx.shape, track_idx.shape, idx)
        #incidence_matrix[part_idx, track_idx] = 1.0

        if self.has_soft_links:
            # Soft incidence: every (node, particle) link of the event with its energy share
            link_start, link_end = self.link_cumsum[idx], self.link_cumsum[idx + 1]
            incidence_matrix = self.compute_incidence_matrix_from_links(
                self.link_node_idx[link_start:link_end],
                self.link_particle_idx[link_start:link_end],
                self.link_weight[link_start:link_end],
                n_nodes,
                n_particles,
            )
        else:
            # Compute incidence matrix ONLY for signal hits (is_noise=False)
            # Get particle assignments for all hits from the pre-stored flat tensor
            part_idx = self.match_indices[topo_start:topo_end]

            # Convert to numpy for efficient computation
            part_idx_np = part_idx.cpu().numpy()

            # Compute incidence matrix (handles signal filtering internally)
            incidence_matrix = self.compute_incidence_matrix(part_idx_np, n_nodes, n_particles)

        if self.verbose:
            print("Computed sparse incidence matrix with shape:", incidence_matrix.shape)

        # Normalize matrix by row (i.e. per-particle normalization) using sklearn's efficient sparse normalization function
        if self.normalize_incidence:
            inplace_csr_row_normalize_l1(incidence_matrix)
            if self.verbose:
                print("Normalized incidence matrix by row (per-particle normalization)")

        # The incidence target must be well defined: finite, within [0, 1], and normalised the
        # way the incidence task expects (energy shares of a node summing to at most 1, or L1
        # rows). This is the matrix that goes into the KL incidence loss and the mask target.
        assert_incidence_matrix(
            incidence_matrix, row_normalized=self.normalize_incidence, context=f"event {self.event_number[idx]}"
        )

        # check for TC w/o associated particles
        #if (incidence_matrix.sum(axis=0) == 0).any():
        #    noisy_cols = np.where(incidence_matrix.sum(axis=0) == 0)[0]
        #    fake_rows = np.arange(len(noisy_cols)) + n_particles
        #    if not (fake_rows < self.num_objects).all():
        #        print(f"Warning: fake_rows go beyond maximum ({self.num_objects}) particles. Dropping them!")
        #        noisy_cols = noisy_cols[fake_rows < self.num_objects]
        #        fake_rows = fake_rows[fake_rows < self.num_objects]
        #    incidence_matrix[fake_rows, noisy_cols] = 1.0

        #normalize
        #incidence_matrix = incidence_matrix.toarray()
        # axis=0 sums over hits (rows), giving per-particle normalization
        #incidence_matrix /= np.clip(incidence_matrix.sum(axis=0, keepdims=True), a_min=1e-6, a_max=None)

        # Convert scipy sparse CSR to COO format, then to PyTorch sparse tensor
        coo = incidence_matrix.tocoo()
        indices = torch.tensor(
            np.array([coo.row, coo.col]), dtype=torch.long
        )
        values = torch.tensor(coo.data, dtype=torch.float32)

        # Create sparse tensor directly at the padded size (no dense padding needed)
        incidence = torch.sparse_coo_tensor(
            indices, values,
            size=(self.num_objects, self.max_nodes)
        ).coalesce()

        # Convert sparse matrix to dense numpy array before converting to PyTorch tensor
        #incidence_matrix_dense = incidence_matrix.toarray()
        
        # Convert the dense incidence matrix to a PyTorch tensor
        #incidence = torch.tensor(incidence_matrix_dense, dtype=torch.float32)

        # Pad: (pad_left, pad_right, pad_top, pad_bottom) for 2D tensor
        # Add padding to bottom (more particles) and right (more nodes)
        #shape_original = incidence.shape
        #incidence = torch.nn.functional.pad(incidence, (0, self.max_nodes - n_nodes, 0, self.num_objects - n_particles))
        if self.verbose:
            print(f"Built sparse incidence matrix with padded shape {incidence.shape} ({n_particles} particles x {n_nodes} nodes before padding)")
        # update the indicator
        is_not_res_mask = particle_class < RESIDUAL_CLASS
        if self.require_nodes_for_valid:
            has_nodes = torch.zeros(n_particles, dtype=torch.bool)
            rows = indices[0]
            has_nodes[rows[rows < n_particles]] = True
            is_not_res_mask = is_not_res_mask & has_nodes
            particle_class = torch.where(is_not_res_mask, particle_class, torch.full_like(particle_class, RESIDUAL_CLASS))
        indicator[:n_particles][is_not_res_mask] = 1.0

        node_inp_features = torch.stack([node_features[f] for f in self.inputs["topocluster"]], dim=-1)

        return {
            "node_inp_features": node_inp_features,
            "node_raw_features": node_raw_features,
            "particle_data": particle_data,
            "incidence_truth": incidence,
            "indicator_truth": indicator,
            "node_q_mask": node_q_mask,
        }

    def __getitem__(self, idx):
        if self.dummy_data:
            return self._generate_dummy_data(idx)

        inputs = {}
        labels = {}

        # load event
        data_dict = self.load_event(idx)

        if self.verbose:
            print(f"Loaded event {idx} with {data_dict['node_inp_features'].shape[0]} nodes and {data_dict['particle_data']['e'].shape[0]} particles")

        inputs = {
            "node_features": data_dict["node_inp_features"],
            "node_valid": data_dict["node_q_mask"],
            "node_e": data_dict["node_raw_features"]["raw_e"],
            "node_pt": data_dict["node_raw_features"]["raw_pt"],
            "node_eta": data_dict["node_raw_features"]["raw_eta"],
            "node_phi": data_dict["node_raw_features"]["raw_phi"],
            "node_sinphi": data_dict["node_raw_features"]["sinphi"],
            "node_cosphi": data_dict["node_raw_features"]["cosphi"],
            "node_r": data_dict["node_raw_features"]["r"],
            "node_is_track": data_dict["node_raw_features"]["is_track"],
            "node_is_noise": data_dict["node_raw_features"]["is_noise"],
            "node_is_pileup": data_dict["node_raw_features"]["is_pileup"],
        }

        if self.has_time:
            inputs["node_time"] = data_dict["node_raw_features"]["raw_time"]
        for name in self.optional_inputs:
            inputs[f"node_{name}"] = data_dict["node_raw_features"][f"raw_{name}"]
        # Node-level proxies of the boundary targets, read by hepattn's regression task as
        # inputs["node_<target>"] (see EXTRA_TARGETS): the raw node position
        for target, canonical in self.proxy_targets.items():
            inputs[f"node_{target}"] = data_dict["node_raw_features"][_PROXY_RAW_KEY[canonical]]

        # set class labels (RESIDUAL_CLASS is residual & fake, 0..NUM_CLASSES-1 are the real classes)
        class_labels = data_dict["particle_data"]["class"].long()
        class_labels[data_dict["indicator_truth"] == 0] = RESIDUAL_CLASS

        labels["particle_class"] = class_labels.long()
        labels["particle_valid"] = data_dict["indicator_truth"].bool()
        labels["node_valid"] = data_dict["node_q_mask"].bool()

        # get masks
        

        # get incidence matrix label
        # labels["particle_incidence"] = data_dict["incidence_truth"]
        
        # get masks — build dense boolean mask from sparse tensor without full dense conversion (otherwise if convert back to dense matrix and then do comparison is not saving anything)
        incidence_sparse = data_dict["incidence_truth"]
        incidence_mask = torch.zeros(incidence_sparse.shape, dtype=torch.bool)
        sparse_idx = incidence_sparse.indices()
        vals = incidence_sparse.values()
        valid = vals > self.incidence_cutval
        incidence_mask[sparse_idx[0][valid], sparse_idx[1][valid]] = True
        incidence_nonzero = torch.zeros(incidence_sparse.shape, dtype=torch.bool)
        nonzero_valid = vals > 0
        incidence_nonzero[sparse_idx[0][nonzero_valid], sparse_idx[1][nonzero_valid]] = True
        incidence_mask_nonzero = incidence_mask & incidence_nonzero
        fraction_masked_nonzero = (1 - incidence_mask_nonzero.sum() / incidence_nonzero.sum()).item()
        # For soft links, entries below the cut are genuine small energy shares rather than
        # lost associations; their fraction is reported once in load_data instead.
        if fraction_masked_nonzero > 0.2 and not self.has_soft_links:
            print(f"⚠️ Warning: High fraction of masked associations in event {idx}: {fraction_masked_nonzero:.2%}. You might consider adjusting the incidence_cutval (currently {self.incidence_cutval}) to lower the number of masked associations.")
            print(f"\tIncidence mask for event {idx}: {incidence_mask_nonzero.sum()} positive associations out of {incidence_nonzero.sum()} non-zero matrix elements ({100 * incidence_mask_nonzero.sum().item() / incidence_nonzero.sum().item():.2f}%)")
        labels["particle_node_valid"] = incidence_mask

        # get incidence matrix label — keep sparse
        labels["particle_incidence"] = incidence_sparse

        # regression targets
        for label in self.targets["particle"]:
            tgt = torch.full((self.num_objects,), torch.nan)  # number of reconstructed tracks
            tgt[: self.n_particles[idx]] = data_dict["particle_data"][label][: self.n_particles[idx]]
            labels[f"particle_{label}"] = tgt

        labels["event_number"] = torch.tensor(self.event_number[idx], dtype=torch.int64)

        return inputs, labels

    def compute_incidence_matrix(self, part_idx_np, n_nodes, n_particles):
        """
        Compute incidence matrix for particle-hit associations using pure numpy.
        Only signal hits (part_idx != -1) contribute to the incidence matrix.
        
        Args:
            part_idx_np: Numpy array of particle indices for each hit (-1 for noise hits)
            n_nodes: Total number of nodes (hits) including noise
            n_particles: Number of particles
        
        Returns:
            Sparse CSR matrix of shape (n_particles, n_nodes)
        """
        # Filter to signal hits only (those with valid particle assignment)
        signal_mask = part_idx_np != -1
        hit_indices = np.where(signal_mask)[0]  # Indices of signal hits
        part_indices = part_idx_np[signal_mask]  # Corresponding particle indices
        
        # Create sparse matrix: incidence[particle_idx, hit_idx] = 1.0
        data = np.ones(len(part_indices), dtype=np.float32)
        
        shape = (n_particles, n_nodes)
        incidence_sparse = csr_matrix((data, (part_indices, hit_indices)), shape=shape, dtype=np.float32)
        
        return incidence_sparse

    def compute_incidence_matrix_from_links(self, node_idx, particle_idx, weight, n_nodes, n_particles):
        """
        Compute a soft incidence matrix from (node, particle, weight) links, with
        incidence[particle_idx, node_idx] = weight, where weight is the fraction of the
        node's energy that belongs to the particle (see _build_soft_links). Nodes without
        links (noise) have an all-zero column. Duplicate (particle, node) pairs, which TICL
        association tables do not produce, would be summed.

        Returns:
            Sparse CSR matrix of shape (n_particles, n_nodes)
        """
        shape = (n_particles, n_nodes)
        return csr_matrix((weight.astype(np.float32), (particle_idx, node_idx)), shape=shape, dtype=np.float32)

    def init_label_dicts(self):
        # hadronic (charged + neutral): 0, electromagnetic (e+- + photon): 1, muon: 2, residual: 3 (= RESIDUAL_CLASS),
        # neutrino: -1, unknown: -2
        self.class_labels = {
            -211: 0,
            211: 0,  # pi+-
            -213: 0,
            213: 0,  # rho+-
            -221: 0,
            221: 0,  # eta+-
            -223: 0,
            223: 0,  # omega(782)
            -321: 0,
            321: 0,  # kaon+-
            -323: 0,
            323: 0,  # K*+-
            -331: 0,
            331: 0,  # eta'(958)
            -333: 0,
            333: 0,  # phi(1020)
            311: 0,  # K0
            -311: 0,
            -411: 0,
            411: 0,  # D+-
            -413: 0,
            413: 0,  # D*(2010)+-
            -423: 0,
            423: 0,  # D*(2007)0
            -431: 0,
            431: 0,  # D_s+-
            -433: 0,
            433: 0,  # D_s*+-
            -511: 0,
            511: 0,  # B0
            -521: 0,
            521: 0,  # B+-
            -523: 0,
            523: 0,  # B*+-
            -531: 0,
            531: 0,  # Bs0
            -541: 0,
            541: 0,  # B_c+-
            -1114: 0,
            1114: 0,  # delta+-
            -2114: 0,
            2114: 0,  # delta0
            -2212: 0,
            2212: 0,  # proton
            -3112: 0,
            3112: 0,  # sigma-
            -3312: 0,
            3312: 0,  # xi+-
            -3222: 0,
            3222: 0,  # sigma+
            -3334: 0,
            3334: 0,  # omega
            -4122: 0,
            4122: 0,  # lambda_c+
            -4132: 0,
            4132: 0,  # xi_c0
            -4232: 0,
            4232: 0,  # xi_c+-
            -4312: 0,
            4312: 0,  # xi'_c0
            -4322: 0,
            4322: 0,  # xi'_c+-
            -4324: 0,
            4324: 0,  # xi*c+-
            -4332: 0,
            4332: 0,  # omega_c0
            -4334: 0,
            4334: 0,  # omega*_c0
            -5112: 0,
            5112: 0,  # lambdab-
            -5122: 0,
            5122: 0,  # lambdab0
            -5132: 0,
            5132: 0,  # xib-
            -5232: 0,
            5232: 0,  # xi0_b
            -5332: 0,
            5332: 0,  # omega_b-
            -11: 1,
            11: 1,  # e
            -13: 2,
            13: 2,  # mu
            -15: 0,
            15: 0,  # tau (calling it hadronic)
            -111: 0,
            111: 0,  # pi0
            113: 0,  # rho0
            130: 0,  # K0L
            310: 0,  # K0S
            -313: 0,
            313: 0,  # K*0
            -421: 0,
            421: 0,  # D0
            -2112: 0,
            2112: 0,  # neutrons
            -3122: 0,
            3122: 0,  # lambda
            -3322: 0,
            3322: 0,  # xi0
            22: 1,  # photon
            1000010020: 0,  # deuteron
            1000010030: 0,  # triton
            1000010040: 0,  # alpha
            1000020030: 0,  # He3
            1000020040: 0,  # He4
            1000030040: 0,  # Li6
            1000030050: 0,  # Li7
            1000020060: 0,  # C6
            1000020070: 0,  # C7
            1000020080: 0,  # O8
            1000010048: 0,  # ?
            1000020032: 0,  # ?
            -999: RESIDUAL_CLASS,  # residual
            -12: -1,
            12: -1,  # nu_e
            -14: -1,
            14: -1,  # nu_mu
            -16: -1,
            16: -1,  # nu_tau
            0: -2,  # unknown
        }

    def init_variables_list(self):
        # Canonical field names are the same regardless of which topocluster/particle
        # collections were chosen — the collection-specific raw branch names are
        # resolved once, in __init__, via TOPO_FIELD_MAP/PARTICLE_FIELD_MAP/
        # PARTICLE_SEED_LOOKUP and MATCH_IDX_MAP/MANY_TO_MANY_MATCH_MAP/ASSOCIATION_MAP.
        self.track_variables = []

        self.topo_variables = [
            'topo_energy',
            'topo_x',
            'topo_y',
            'topo_z',
            'topo_eta',
            'topo_phi',
            'topo_cosphi',
            'topo_sinphi',
            'topo_R',
            'topo_is_noise',
            'topo_is_pileup',
            'topo_match_idx',  # Needed for incidence matrix computation
        ]

        if self.has_time:
            self.topo_variables += [
                'topo_time'
            ]

        self.topo_variables += [f"topo_{canonical}" for canonical in self.optional_inputs.values()]

        self.particle_variables = [
            "particle_pdgId",
            "particle_eta",
            "particle_phi",
            "particle_x",
            "particle_y",
            "particle_z",
            "particle_R",
            "particle_recEnergy",
        ]
        # Boundary fields of the seed particle (entry point into HGCAL and momentum there),
        # when the collection provides them; the entrance-position targets are built from
        # them (EXTRA_TARGETS). Missing branches are dropped again in __init__.
        self.particle_variables += [f"particle_{canonical}" for canonical in self.boundary_fields]

        self.aux_vars = [
        ]

class HGCDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        scale_dict_path: str,
        particle_scale_dict_path: str,
        train_path: str | None = None,
        valid_path: str | None = None,
        test_path: str | None = None,
        data_dir: str | None = None,
        split: list[int | float] | None = None,
        split_seed: int | None = 42,
        pin_memory: bool = True,
        test_suff: str | None = None,
        **kwargs,
    ):
        """Each split reads either its own path (`train_path`/`valid_path`/`test_path`: a parquet
        file or a directory of them) or its share of `data_dir`, cut into whole files by `split`
        ([train, val, test] as file counts or fractions, see split_parquet_files) after shuffling
        the file list with `split_seed`. An explicit path overrides the split for that split, e.g.
        --data.test_path to evaluate a data_dir-trained model on another file."""
        super().__init__()

        paths = {"train": train_path, "val": valid_path, "test": test_path}
        if data_dir is not None:
            if split is None:
                raise ValueError("data_dir needs a split, e.g. split: [0.8, 0.1, 0.1]")
            split_files = split_parquet_files(data_dir, split, seed=split_seed)
            print(f"Split the {sum(map(len, split_files.values()))} files used from {data_dir} into "
                  + ", ".join(f"{name} {len(files)}" for name, files in split_files.items())
                  + f" (split={split}, split_seed={split_seed})")
            for name, files in split_files.items():
                if paths[name] is not None:
                    print(f"{name}: explicit path {paths[name]} overrides the {len(files)} {name} files of data_dir")
                elif files:
                    paths[name] = [str(f) for f in files]
        elif split is not None:
            raise ValueError("split is only used together with data_dir")

        self.train_path = paths["train"]
        self.valid_path = paths["val"]
        self.batch_size = batch_size
        self.test_path = paths["test"]
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.test_suff = test_suff
        self.scale_dict_path = scale_dict_path
        self.particle_scale_dict_path = particle_scale_dict_path
        self.kwargs = kwargs

    def setup(self, stage: str):
        if self.trainer.is_global_zero:
            print("-" * 100)

        # create training and validation datasets
        if stage == "fit":
            assert self.train_path is not None and self.valid_path is not None, \
                "No train/validation files, see --data.train_path/--data.valid_path or --data.data_dir/--data.split"
            self.train_dset = HGCDataset(
                filepath=self.train_path,
                num_events=self.num_train,
                scale_dict_path=self.scale_dict_path,
                particle_scale_dict_path=self.particle_scale_dict_path,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dset = HGCDataset(
                filepath=self.valid_path,
                num_events=self.num_val,
                scale_dict_path=self.scale_dict_path,
                particle_scale_dict_path=self.particle_scale_dict_path,
                **self.kwargs,
            )

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_path is not None, "No test file specified, see --data.test_path or --data.data_dir/--data.split"
            self.test_dset = HGCDataset(
                filepath=self.test_path,
                num_events=self.num_test,
                scale_dict_path=self.scale_dict_path,
                particle_scale_dict_path=self.particle_scale_dict_path,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dset):,} events")

        if stage == "fit":
            self.write_dataset_counts({"train": self.train_dset, "val": self.val_dset})
        elif stage == "test":
            self.write_dataset_counts({"test": self.test_dset})

        if self.trainer.is_global_zero:
            print("-" * 100, "\n")

    def write_dataset_counts(self, datasets: dict):
        """Write each dataset's event counts and cutflow (HGCDataset.counts) to
        <trainer.log_dir>/dataset_counts.json, keyed by split. Splits from an earlier stage
        already in the file (e.g. train/val when testing) are kept."""
        log_dir = self.trainer.log_dir  # broadcasts the directory, so call it on every rank
        counts = {split: dset.counts for split, dset in datasets.items() if hasattr(dset, "counts")}
        if not self.trainer.is_global_zero or not counts:
            return
        if log_dir is None:
            print("No trainer log_dir, dataset_counts.json not written")
            return
        path = Path(log_dir) / "dataset_counts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            counts = {**json.loads(path.read_text()), **counts}
        path.write_text(json.dumps(counts, indent=2))
        print(f"Saved dataset counts and cutflow to {path}")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        inputs, targets = batch
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        for k, v in targets.items():
            if isinstance(v, list) and len(v) > 0 and v[0].is_sparse:
                targets[k] = torch.stack([x.to_dense() for x in v]).to(device)
            elif isinstance(v, torch.Tensor):
                targets[k] = v.to(device)
        return inputs, targets

    def get_dataloader(self, stage: str, dataset: HGCDataset, shuffle: bool):
        print(f"Creating {stage} dataloader with {len(dataset):,} events")
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=sparse_collate_fn,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        print("Instantiating train dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        print("Instantiating validation dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        print("Instantiating test dataloader on rank", self.trainer.local_rank)
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)
