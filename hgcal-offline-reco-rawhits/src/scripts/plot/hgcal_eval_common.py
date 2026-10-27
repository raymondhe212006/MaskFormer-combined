"""Shared helpers for evaluating the HGCAL MaskFormer predictions.

The prediction files are the awkward/parquet dumps written by
``callbacks.prediction_writer.HGCPredictionWriter``: one record per event, the full schema
of the input file, plus prediction branches on two collections:

the *particle* collection (``SimCluster``, ``MergedSimCluster`` or ``ticlSimTracksters``)
    One entry per true particle.  The ``<particle>_pred_*`` fields hold the model output for
    the query that the Hungarian matcher assigned to that particle, so index ``i`` of a
    ``pred`` field and index ``i`` of a truth field describe the same object.

the *node* (topocluster) collection (``RecHitHGC``, ``LayerCluster`` or
``ticlTrackstersCLUE3DHigh``)
    One entry per input node.  ``<node>_pred_match_idx`` holds the particle index the model
    assigned the node to (``-1`` = no query predicts it, ``-2`` = an unmatched/extra query).

Which raw branches hold the kinematics, and how the truth node -> particle association is
stored, is taken from ``dataset.collections`` (the same tables the training dataset uses):

* ``best``   -- a per-node best-match index/quality pair (``RecHitHGC_*BestMatchIdx/Qual``);
* ``ragged`` -- a per-node list of candidates (``LayerCluster_SimCluster_MatchIdx/MatchQual``,
  ``LayerCluster_SimClusterNumMatch`` entries each), reduced with the direction recorded in
  ``MANY_TO_MANY_MATCH_MAP`` (``MatchQual`` is a score: the smallest value wins);
* ``links``  -- a TICL association table (``CLUE3DHighToSimTSByHits``...): per node the
  particles it shares energy with; the best match is the one with the largest share of the
  node's energy, as in ``HGCDataset._build_soft_links``.

The two collections are auto-detected from the parquet schema (the ones carrying
``*_pred_match_idx`` and ``*_pred_valid``) unless given explicitly.
"""

import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np

# `dataset.collections` is torch-free; make it importable whether this file is run from the
# repo checkout, inside the container (PYTHONPATH=src) or through run_plot_chain.py
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from dataset.caloparticle_seeds import resolve_caloparticle_seeds  # noqa: E402
from dataset.collections import (  # noqa: E402
    ASSOCIATION_MAP, BOUNDARY_FIELDS, MANY_TO_MANY_MATCH_MAP, MATCH_IDX_MAP, PARTICLE_FIELD_MAP,
    PARTICLE_SEED_LOOKUP, TOPO_FIELD_MAP, association_fields, boundary_eta_phi, supported_pairs,
)

#: the entrance-position regression targets (EXTRA_TARGETS of dataset_maskformers, which the
#: chain must not import): built from the boundary fields of the particle collection, the
#: point where the true particle crossed the HGCAL front face
BOUNDARY_TARGETS = ("boundary_x", "boundary_y", "boundary_z", "boundary_eta", "boundary_sinphi", "boundary_cosphi")

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #

CMAP_10 = [
    "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
    "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd",
]
CMS_blue, CMS_orange, CMS_red, CMS_gray, CMS_purple = CMAP_10[:5]

_STYLE_ANNOUNCED = False


def setup_style():
    """Configure matplotlib (mplhep CMS style when available) and return ``pyplot``."""
    global _STYLE_ANNOUNCED
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import mplhep as hep
        hep.style.use("CMS")
        style = "mplhep CMS style"
    except ImportError:
        # the training container has matplotlib but no mplhep: approximate the CMS look
        plt.rcParams.update({
            "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18,
            "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
            "ytick.minor.visible": True, "axes.linewidth": 1.2, "lines.linewidth": 2,
            "legend.frameon": False, "mathtext.fontset": "dejavusans",
        })
        style = "plain matplotlib style (mplhep not installed)"
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    if not _STYLE_ANNOUNCED:
        print("[style] %s" % style)
        _STYLE_ANNOUNCED = True
    return plt


# --------------------------------------------------------------------------- #
# PDG -> class
# --------------------------------------------------------------------------- #

#: Must match HGCDataset.init_label_dicts: hadronic (charged + neutral) 0, electromagnetic
#: (e+- + photon) 1, muon 2, residual 3, neutrino -1, unknown -2. The residual class is the
#: model's null class, at index num_classes (3) in hepattn's ObjectClassificationTask.
RESIDUAL_CLASS = 3

class_labels = {
    -211: 0, 211: 0, -213: 0, 213: 0, -221: 0, 221: 0, -223: 0, 223: 0,
    -321: 0, 321: 0, -323: 0, 323: 0, -331: 0, 331: 0, -333: 0, 333: 0,
    311: 0, -311: 0, -411: 0, 411: 0, -413: 0, 413: 0, -423: 0, 423: 0,
    -431: 0, 431: 0, -433: 0, 433: 0, -511: 0, 511: 0, -521: 0, 521: 0,
    -523: 0, 523: 0, -531: 0, 531: 0, -541: 0, 541: 0, -1114: 0, 1114: 0,
    -2114: 0, 2114: 0, -2212: 0, 2212: 0, -3112: 0, 3112: 0, -3312: 0,
    3312: 0, -3222: 0, 3222: 0, -3334: 0, 3334: 0, -4122: 0, 4122: 0,
    -4132: 0, 4132: 0, -4232: 0, 4232: 0, -4312: 0, 4312: 0, -4322: 0,
    4322: 0, -4324: 0, 4324: 0, -4332: 0, 4332: 0, -4334: 0, 4334: 0,
    -5112: 0, 5112: 0, -5122: 0, 5122: 0, -5132: 0, 5132: 0, -5232: 0,
    5232: 0, -5332: 0, 5332: 0, -11: 1, 11: 1, -13: 2, 13: 2, -15: 0,
    15: 0, -111: 0, 111: 0, 113: 0, 130: 0, 310: 0, -313: 0, 313: 0,
    -421: 0, 421: 0, -2112: 0, 2112: 0, -3122: 0, 3122: 0, -3322: 0,
    3322: 0, 22: 1,
    1000010020: 0, 1000010030: 0, 1000010040: 0, 1000020030: 0,
    1000020040: 0, 1000030040: 0, 1000030050: 0, 1000020060: 0,
    1000020070: 0, 1000020080: 0, 1000010048: 0, 1000020032: 0,
    -999: RESIDUAL_CLASS, -12: -1, 12: -1, -14: -1, 14: -1, -16: -1, 16: -1, 0: -2,
}

_keys = np.array(sorted(class_labels), dtype=np.int64)
_vals = np.array([class_labels[k] for k in _keys], dtype=np.int8)
UNMAPPED = np.int8(-2)

#: class index -> (name, colour).  RESIDUAL_CLASS is the model's "null / no object" class.
CLASS_INFO = {
    0: ("hadronic", CMAP_10[0]),
    1: ("electromagnetic", CMAP_10[1]),
    2: ("muon", CMAP_10[2]),
    RESIDUAL_CLASS: ("residual / null", CMAP_10[3]),
}
PHYSICS_CLASSES = [0, 1, 2]


def pdg_to_class(flat):
    flat = np.asarray(flat, dtype=np.int64)
    i = np.searchsorted(_keys, flat)
    i_clip = np.clip(i, 0, len(_keys) - 1)
    hit = _keys[i_clip] == flat
    return np.where(hit, _vals[i_clip], UNMAPPED)


def class_name(c):
    return CLASS_INFO.get(int(c), (f"class {int(c)}", CMS_gray))[0]


def class_color(c):
    return CLASS_INFO.get(int(c), (f"class {int(c)}", CMS_gray))[1]


# --------------------------------------------------------------------------- #
# collections: labels, schema, pair spec
# --------------------------------------------------------------------------- #

#: plot labels per node collection: (long label, short label for formulas)
NODE_EXTRAS = {
    "RecHitHGC": dict(label="RecHit", short="hit"),
    "LayerCluster": dict(label="LayerCluster", short="LC"),
    "ticlTrackstersCLUE3DHigh": dict(label="trackster", short="TS"),
}

#: plot-only knowledge per particle collection: label, an alternative energy definition
#: to compare the prediction against, and a grouping index (CaloParticle) for the
#: sample-structure plot.  ``group=("seed", branch)`` reads the branch off the seed
#: collection (see PARTICLE_SEED_LOOKUP).  Branches missing from the file are skipped.
PARTICLE_EXTRAS = {
    "SimCluster": dict(label="SimCluster", alt_energy="SimCluster_boundaryEnergy",
                       alt_label="boundaryEnergy", group="SimCluster_CaloPartIdx"),
    "MergedSimCluster": dict(label="MergedSimCluster", alt_energy="MergedSimCluster_boundaryEnergy",
                             alt_label="boundaryEnergy", group=None),
    "ticlSimTracksters": dict(label="SimTrackster", alt_energy="ticlSimTracksters_raw_energy",
                              alt_label="raw_energy", group=("seed", "SimCluster_CaloPartIdx")),
}

PRED_BOOKKEEPING = ("class", "valid", "n_nodes")


def read_schema(path):
    """``{top-level collection: set(field names)}`` from the parquet schema (no data read)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    out = {}
    for name in schema.names:
        t = schema.field(name).type
        while pa.types.is_list(t) or pa.types.is_large_list(t):
            t = t.value_type
        if pa.types.is_struct(t):
            out[name] = {t.field(i).name for i in range(t.num_fields)}
        else:
            out[name] = set()
    return out


def _pick(requested, found, kind, marker, path):
    if requested is not None:
        if requested not in found:
            raise ValueError("%s: %s collection %r carries no %s (collections with one: %s)"
                             % (os.path.basename(path), kind, requested, marker,
                                ", ".join(found) or "none"))
        return requested
    if not found:
        raise ValueError("%s: no collection carries a %s branch; is this a prediction file "
                         "written by HGCPredictionWriter?" % (os.path.basename(path), marker))
    if len(found) > 1:
        raise ValueError("%s: several %s collections carry %s (%s); pass --%s-collection"
                         % (os.path.basename(path), kind, marker, ", ".join(found), kind))
    return found[0]


def detect_pair(path, topocluster_collection=None, particle_collection=None, schema=None):
    """Return ``(topocluster_collection, particle_collection)`` of a prediction file.

    The node collection is the one carrying ``<coll>_pred_match_idx``, the particle
    collection the one carrying ``<coll>_pred_valid``; explicit choices are validated.
    """
    schema = read_schema(path) if schema is None else schema
    topo_found = sorted(c for c, f in schema.items() if "%s_pred_match_idx" % c in f)
    part_found = sorted(c for c, f in schema.items() if "%s_pred_valid" % c in f)
    topo = _pick(topocluster_collection, topo_found, "topocluster", "*_pred_match_idx", path)
    part = _pick(particle_collection, part_found, "particle", "*_pred_valid", path)
    if (topo, part) not in supported_pairs():
        raise ValueError("(%s, %s) is not a supported (topocluster, particle) pair; known: %s"
                         % (topo, part, supported_pairs()))
    return topo, part


def build_spec(topocluster_collection, particle_collection, schema,
               association_collection=None, min_link_fraction=0.0):
    """Branch names and truth mode for one (topocluster, particle) pair, checked
    against the file schema.  Returns a plain dict (see EvalData for its use)."""
    topo, part = topocluster_collection, particle_collection
    pair = (topo, part)
    if pair not in supported_pairs():
        raise ValueError("unsupported pair %s; known: %s" % (pair, supported_pairs()))
    topo_fields = schema.get(topo, set())
    part_fields = schema.get(part, set())

    def need(coll, fields, branch, what):
        if branch not in fields:
            raise ValueError("collection %r has no branch %r (%s); has: %s"
                             % (coll, branch, what, ", ".join(sorted(fields))))
        return branch

    tf = TOPO_FIELD_MAP[topo]
    node = dict(
        energy=need(topo, topo_fields, tf["energy"], "node energy"),
        x=need(topo, topo_fields, tf["x"], "node x"),
        y=need(topo, topo_fields, tf["y"], "node y"),
        z=need(topo, topo_fields, tf["z"], "node z"),
        eta=tf.get("eta") if tf.get("eta") in topo_fields else None,
        phi=tf.get("phi") if tf.get("phi") in topo_fields else None,
        pred_match_idx=need(topo, topo_fields, "%s_pred_match_idx" % topo, "prediction"),
    )

    pf = PARTICLE_FIELD_MAP[part]
    particle = dict(
        energy=need(part, part_fields, pf["recEnergy"], "energy target"),
        eta=need(part, part_fields, pf["eta"], "eta"),
        phi=need(part, part_fields, pf["phi"], "phi"),
        x=need(part, part_fields, pf["x"], "x"),
        y=need(part, part_fields, pf["y"], "y"),
        z=need(part, part_fields, pf["z"], "z"),
        pdgId=pf.get("pdgId") if pf.get("pdgId") in part_fields else None,
    )
    # optional: the seed particle at the HGCAL boundary (entry point and momentum there, the
    # truth of the entrance-position targets); None when the collection or the file lacks it
    for canonical in BOUNDARY_FIELDS:
        branch = pf.get(canonical)
        particle[canonical] = branch if branch in part_fields else None
    extras = PARTICLE_EXTRAS.get(part, {})
    alt_energy = extras.get("alt_energy")
    if alt_energy not in part_fields:
        alt_energy = None
    group = extras.get("group")

    seed = None
    if part in PARTICLE_SEED_LOOKUP:
        lk = PARTICLE_SEED_LOOKUP[part]
        seed_coll = lk["seed_collection"]
        seed_fields = schema.get(seed_coll)
        if seed_fields is None:
            raise ValueError("particle collection %r takes its pdgId from %r, which is not in the file"
                             % (part, seed_coll))
        cp = lk.get("caloparticle_seed") or {}
        seed = dict(
            collection=seed_coll,
            idx=need(part, part_fields, lk["seed_idx"], "seed index"),
            flag=lk.get("seed_is_caloparticle") if lk.get("seed_is_caloparticle") in part_fields else None,
            pdgId=need(seed_coll, seed_fields, lk["fields"]["pdgId"], "seed pdgId"),
            group=None,
            # CaloParticle-seeded particles (an unconverted photon) are resolved to a seed of
            # their CaloParticle through these branches, see dataset.caloparticle_seeds
            cp_group=cp.get("group_idx") if cp.get("group_idx") in seed_fields else None,
            cp_energy=cp.get("energy") if cp.get("energy") in seed_fields else None,
        )
        if isinstance(group, tuple) and group[0] == "seed":
            seed["group"] = group[1] if group[1] in seed_fields else None
        group = None
    elif particle["pdgId"] is None:
        raise ValueError("particle collection %r provides no pdgId branch" % part)
    else:
        if not isinstance(group, str) or group not in part_fields:
            group = None

    if pair in ASSOCIATION_MAP:
        table = association_collection or ASSOCIATION_MAP[pair]["default"]
        if table not in ASSOCIATION_MAP[pair]["choices"]:
            raise ValueError("association table %r does not describe %s; known: %s"
                             % (table, pair, list(ASSOCIATION_MAP[pair]["choices"])))
        fields = association_fields(table)
        table_fields = schema.get(table)
        if table_fields is None:
            raise ValueError("association table %r is not in the file" % table)
        for b in fields.values():
            need(table, table_fields, b, "association link")
        truth = dict(mode="links", table=table, fields=fields,
                     min_link_fraction=float(min_link_fraction))
    elif association_collection is not None:
        raise ValueError("association_collection is only used by %s; %s stores a per-node "
                         "match index" % (sorted(ASSOCIATION_MAP), pair))
    elif pair in MANY_TO_MANY_MATCH_MAP:
        m = MANY_TO_MANY_MATCH_MAP[pair]
        truth = dict(mode="ragged",
                     num_match=need(topo, topo_fields, m["num_match"], "truth candidates"),
                     idx=need(topo, topo_fields, m["match_idx"], "truth candidates"),
                     qual=need(topo, topo_fields, m["match_qual"], "truth candidates"),
                     best=m.get("best", "min"))
    else:
        idx = need(topo, topo_fields, MATCH_IDX_MAP[pair], "truth best match")
        qual = idx[:-3] + "Qual" if idx.endswith("Idx") else None
        truth = dict(mode="best", idx=idx, qual=qual if qual in topo_fields else None)

    pred_prefix = "%s_pred_" % part
    pred = sorted(f[len(pred_prefix):] for f in part_fields if f.startswith(pred_prefix))
    for b in PRED_BOOKKEEPING:
        if b not in pred:
            raise ValueError("collection %r has no %s%s branch" % (part, pred_prefix, b))
    regressed = [f for f in pred if f not in PRED_BOOKKEEPING]

    node_extra = NODE_EXTRAS.get(topo, dict(label=topo, short=topo))
    return dict(
        topocluster_collection=topo, particle_collection=part,
        node=node, particle=particle, alt_energy=alt_energy,
        alt_label=extras.get("alt_label") if alt_energy else None,
        group=group, seed=seed, truth=truth, regressed=regressed,
        node_label=node_extra["label"], node_short=node_extra["short"],
        particle_label=extras.get("label", part),
        energy_label=particle["energy"][len(part) + 1:],
    )


def columns_for(spec):
    """awkward ``from_parquet`` column patterns reading only what EvalData needs."""
    topo, part = spec["topocluster_collection"], spec["particle_collection"]
    cols = ["%s.%s" % (topo, b) for b in spec["node"].values() if b]
    t = spec["truth"]
    if t["mode"] == "best":
        cols += ["%s.%s" % (topo, b) for b in (t["idx"], t["qual"]) if b]
    elif t["mode"] == "ragged":
        cols += ["%s.%s" % (topo, t[k]) for k in ("num_match", "idx", "qual")]
    else:
        cols += ["%s.%s" % (t["table"], b) for b in t["fields"].values()]
    cols += ["%s.%s" % (part, b) for b in spec["particle"].values() if b]
    if spec["alt_energy"]:
        cols.append("%s.%s" % (part, spec["alt_energy"]))
    if spec["group"]:
        cols.append("%s.%s" % (part, spec["group"]))
    cols += ["%s.%s_pred_%s" % (part, part, f) for f in list(PRED_BOOKKEEPING) + spec["regressed"]]
    seed = spec["seed"]
    if seed:
        cols.append("%s.%s" % (part, seed["idx"]))
        if seed["flag"]:
            cols.append("%s.%s" % (part, seed["flag"]))
        cols.append("%s.%s" % (seed["collection"], seed["pdgId"]))
        if seed["group"]:
            cols.append("%s.%s" % (seed["collection"], seed["group"]))
        for b in (seed.get("cp_group"), seed.get("cp_energy")):
            if b:
                cols.append("%s.%s" % (seed["collection"], b))
    return sorted(set(cols))


# --------------------------------------------------------------------------- #
# loading / flattening
# --------------------------------------------------------------------------- #

def _f(arr):
    """Flatten a jagged array to a contiguous numpy array."""
    return np.asarray(ak.to_numpy(ak.flatten(arr)))


def _offsets(counts):
    off = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    return off


def _segment_argbest(seg, idx, key, n_segments, prefer="min"):
    """Per segment, the ``idx`` whose ``key`` is smallest (``prefer="min"``) or largest.

    ``seg`` is the segment id of every entry (entries of one segment need not be
    contiguous).  Empty segments get ``-1`` / ``nan``.
    """
    best = np.full(n_segments, -1, dtype=np.int64)
    best_key = np.full(n_segments, np.nan)
    if len(seg) == 0:
        return best, best_key
    k = key if prefer == "min" else -key
    order = np.lexsort((k, seg))
    seg_s = seg[order]
    first = np.empty(len(seg_s), dtype=bool)
    first[0] = True
    first[1:] = seg_s[1:] != seg_s[:-1]
    best[seg_s[first]] = idx[order][first]
    best_key[seg_s[first]] = key[order][first]
    return best, best_key


class EvalData:
    """Flat (event-concatenated) view of one prediction file.

    Attributes with the ``p_`` prefix are per *particle*, those with the ``n_`` prefix are
    per *node*.  Node->particle pointers (``n_pred``, ``n_true``) are stored as **global**
    indices into the particle arrays (``-1`` = none) so that everything can be done with
    plain numpy.  Which collections and branches are read is described by ``self.spec``
    (see ``build_spec``); labels for plots are ``particle_label``, ``node_label``,
    ``node_short``, ``energy_label`` and ``energy_alt_label``.
    """

    def __init__(self, path, max_events=None, topocluster_collection=None,
                 particle_collection=None, association_collection=None,
                 min_link_fraction=0.0):
        self.path = path
        schema = read_schema(path)
        topo, part = detect_pair(path, topocluster_collection, particle_collection, schema=schema)
        spec = build_spec(topo, part, schema, association_collection, min_link_fraction)
        self.spec = spec
        self.topocluster_collection = topo
        self.particle_collection = part
        self.node_label = spec["node_label"]
        self.node_short = spec["node_short"]
        self.particle_label = spec["particle_label"]
        self.energy_label = spec["energy_label"]
        self.energy_alt_label = spec["alt_label"]
        self.truth_mode = spec["truth"]["mode"]
        self.regressed_fields = list(spec["regressed"])

        # only the branches that matter: a RecHit file is ~20M nodes and the other
        # collections/association tables would multiply the memory for nothing
        df = ak.from_parquet(path, columns=columns_for(spec))
        if max_events is not None:
            df = df[:max_events]
        pc, nc = df[part], df[topo]
        ps, ns = spec["particle"], spec["node"]

        self.n_events = len(df)
        self.p_count = np.asarray(ak.to_numpy(ak.num(pc[ps["energy"]], axis=1)))
        self.n_count = np.asarray(ak.to_numpy(ak.num(nc[ns["energy"]], axis=1)))
        self.p_off = _offsets(self.p_count)
        self.n_off = _offsets(self.n_count)
        self.p_event = np.repeat(np.arange(self.n_events), self.p_count)
        self.n_event = np.repeat(np.arange(self.n_events), self.n_count)

        # ---------------- particle level: truth ----------------
        self.p_e = _f(pc[ps["energy"]]).astype(np.float64)
        self.p_e_alt = _f(pc[spec["alt_energy"]]).astype(np.float64) if spec["alt_energy"] else None
        self.p_eta = _f(pc[ps["eta"]]).astype(np.float64)
        self.p_phi = _f(pc[ps["phi"]]).astype(np.float64)
        self.p_x = _f(pc[ps["x"]]).astype(np.float64)
        self.p_y = _f(pc[ps["y"]]).astype(np.float64)
        self.p_z = _f(pc[ps["z"]]).astype(np.float64)
        self.p_sinphi = np.sin(self.p_phi)
        self.p_cosphi = np.cos(self.p_phi)
        # pred_r matches the 3D radius of the position (~300 cm), not the transverse one
        self.p_r = np.sqrt(self.p_x ** 2 + self.p_y ** 2 + self.p_z ** 2)
        # entry point of the true particle into HGCAL (and its momentum there): the truth
        # of the boundary targets, None where the file has no such branch
        for canonical in BOUNDARY_FIELDS:
            branch = ps.get(canonical)
            setattr(self, "p_%s" % canonical, _f(pc[branch]).astype(np.float64) if branch else None)
        # eta/phi of the entry point from its position, as the dataset does (the dumped
        # boundaryPhi is +-eta in the CMSSW_20_0_0_pre1 production, see dataset.collections)
        if self.has_boundary_truth and self.p_boundary_z is not None:
            self.p_boundary_eta, self.p_boundary_phi = boundary_eta_phi(
                self.p_boundary_x, self.p_boundary_y, self.p_boundary_z)
        has_bphi = self.p_boundary_phi is not None
        self.p_boundary_sinphi = np.sin(self.p_boundary_phi) if has_bphi else None
        self.p_boundary_cosphi = np.cos(self.p_boundary_phi) if has_bphi else None

        group_local = None
        seed = spec["seed"]
        if seed is not None:
            seeds = df[seed["collection"]]
            seed_idx = pc[seed["idx"]]
            unmapped = None
            if seed["flag"] is not None and ak.any(pc[seed["flag"]]):
                # Seeded by the CaloParticle itself (it entered HGCAL intact, e.g. an unconverted
                # photon): seed_idx is a CaloParticle row. Resolve it to the most energetic seed
                # of that CaloParticle (dataset.caloparticle_seeds). SimCluster-seeded particles
                # (every tau decay product) are untouched.
                is_cp = pc[seed["flag"]]
                n_cp = int(ak.sum(is_cp))
                if seed["cp_group"] is not None:
                    energy = seeds[seed["cp_energy"]] if seed["cp_energy"] else None
                    seed_idx, n_unresolved = resolve_caloparticle_seeds(
                        seed_idx, is_cp, seeds[seed["cp_group"]], energy)
                    if n_unresolved:
                        raise ValueError("%d CaloParticle-seeded %s particles have no %s with %s equal to their %s"
                                         % (n_unresolved, part, seed["collection"], seed["cp_group"], seed["idx"]))
                    print("[eval] %d %s particles are seeded by a CaloParticle (it entered HGCAL intact); "
                          "their pdgId is that of the most energetic %s of the CaloParticle"
                          % (n_cp, part, seed["collection"]))
                else:
                    print("[eval] WARNING: %d %s particles are seeded by a CaloParticle and %s carries no "
                          "CaloParticle row branch to resolve them; their pdgId is unknown (class set to "
                          "unmapped)" % (n_cp, part, seed["collection"]))
                    unmapped = _f(is_cp).astype(bool)
                    seed_idx = ak.where(is_cp, 0, seed_idx)   # a CaloParticle row is no seed row
            n_seeds = ak.num(seeds[seed["pdgId"]], axis=1)
            if ak.any(seed_idx < 0) or ak.any(seed_idx >= n_seeds):
                raise ValueError("%s points outside the %r collection" % (seed["idx"], seed["collection"]))
            pdg = _f(seeds[seed["pdgId"]][seed_idx]).astype(np.int64)
            if unmapped is not None:
                pdg = np.where(unmapped, 0, pdg)
            if seed["group"]:
                group_local = _f(seeds[seed["group"]][seed_idx]).astype(np.int64)
        else:
            pdg = _f(pc[ps["pdgId"]]).astype(np.int64)
            if spec["group"]:
                group_local = _f(pc[spec["group"]]).astype(np.int64)
        self.p_pdg = pdg
        self.p_class = pdg_to_class(self.p_pdg).astype(np.int64)
        #: global id of the group (CaloParticle) each particle belongs to, or None
        self.p_group = None
        if group_local is not None and len(group_local):
            stride = int(max(group_local.max(), 0)) + 1
            self.p_group = self.p_event.astype(np.int64) * stride + group_local

        # ---------------- particle level: prediction ----------------
        for fld in self.regressed_fields:
            setattr(self, "p_pred_%s" % fld, _f(pc["%s_pred_%s" % (part, fld)]).astype(np.float64))
        for fld in ("e", "eta", "sinphi", "cosphi", "r", "phi", "boundary_phi") + BOUNDARY_TARGETS:
            if not hasattr(self, "p_pred_%s" % fld):
                setattr(self, "p_pred_%s" % fld, None)
        if self.p_pred_phi is None and self.p_pred_sinphi is not None and self.p_pred_cosphi is not None:
            self.p_pred_phi = np.arctan2(self.p_pred_sinphi, self.p_pred_cosphi)
        if (self.p_pred_boundary_phi is None and self.p_pred_boundary_sinphi is not None
                and self.p_pred_boundary_cosphi is not None):
            self.p_pred_boundary_phi = np.arctan2(self.p_pred_boundary_sinphi, self.p_pred_boundary_cosphi)
        self.p_pred_class = _f(pc["%s_pred_class" % part]).astype(np.int64)
        self.p_valid = _f(pc["%s_pred_valid" % part]).astype(bool)
        self.p_pred_n_nodes = _f(pc["%s_pred_n_nodes" % part]).astype(np.int64)

        # ---------------- node level ----------------
        self.n_energy = _f(nc[ns["energy"]]).astype(np.float64)
        # the cartesian position is kept for every pair (event displays, node 3D radius);
        # anything else per node is derived on demand by node_feature(), a RecHit file
        # having ~20M nodes
        self.n_x = _f(nc[ns["x"]]).astype(np.float64)
        self.n_y = _f(nc[ns["y"]]).astype(np.float64)
        self.n_z = _f(nc[ns["z"]]).astype(np.float64)
        if ns["eta"] and ns["phi"]:
            self.n_eta = _f(nc[ns["eta"]]).astype(np.float64)
            self.n_phi = _f(nc[ns["phi"]]).astype(np.float64)
        else:
            # RecHits only store the cartesian position
            rt = np.hypot(self.n_x, self.n_y)
            self.n_eta = np.arcsinh(np.divide(self.n_z, rt, out=np.zeros_like(rt), where=rt > 0))
            self.n_phi = np.arctan2(self.n_y, self.n_x)

        n_pred_local = _f(nc[ns["pred_match_idx"]]).astype(np.int64)
        n_true_local, n_true_qual = self._truth_match(df, nc, spec["truth"])
        self.n_pred_local = n_pred_local
        self.n_true_local = n_true_local
        self.n_true_qual = n_true_qual

        base = self.p_off[self.n_event]
        self.n_pred = np.where(n_pred_local >= 0, base + n_pred_local, -1)
        self.n_true = np.where(n_true_local >= 0, base + n_true_local, -1)

        # per-particle node bookkeeping derived from the assignments
        n_part = len(self.p_e)
        self.p_n_true_nodes = np.bincount(self.n_true[self.n_true >= 0], minlength=n_part)
        self.p_n_pred_nodes = np.bincount(self.n_pred[self.n_pred >= 0], minlength=n_part)
        self.p_sum_e_true_nodes = np.bincount(
            self.n_true[self.n_true >= 0],
            weights=self.n_energy[self.n_true >= 0], minlength=n_part)
        self.p_sum_e_pred_nodes = np.bincount(
            self.n_pred[self.n_pred >= 0],
            weights=self.n_energy[self.n_pred >= 0], minlength=n_part)

    # -------------------------------------------------------------- #
    def _truth_match(self, df, nc, truth):
        """Per node: the (event-local) index of the truth-associated particle, ``-1`` if
        none, and the quality of that association (``nan`` if none)."""
        n_nodes = len(self.n_energy)
        if truth["mode"] == "best":
            idx = _f(nc[truth["idx"]]).astype(np.int64)
            idx = np.where(idx >= 0, idx, -1)
            qual = _f(nc[truth["qual"]]).astype(np.float64) if truth["qual"] else np.full(n_nodes, np.nan)
            return idx, np.where(idx >= 0, qual, np.nan)

        if truth["mode"] == "ragged":
            nmatch = _f(nc[truth["num_match"]]).astype(np.int64)
            idx = _f(nc[truth["idx"]]).astype(np.int64)
            qual = _f(nc[truth["qual"]]).astype(np.float64)
            if nmatch.sum() != len(idx):
                raise ValueError("%s does not add up to the length of %s" % (truth["num_match"], truth["idx"]))
            seg = np.repeat(np.arange(n_nodes), nmatch)
            return _segment_argbest(seg, idx, qual, n_nodes, prefer=truth["best"])

        # links: TICL association table, one row per node, links flattened per event in
        # node order; the best match is the particle with the largest share of the node
        # energy (HGCDataset._build_soft_links)
        fields = truth["fields"]
        table = df[truth["table"]]
        n_links = _f(table[fields["n_links"]]).astype(np.int64)
        if len(n_links) != n_nodes:
            raise ValueError("association table %r has %d rows for %d nodes"
                             % (truth["table"], len(n_links), n_nodes))
        idx = _f(table[fields["index"]]).astype(np.int64)
        shared = _f(table[fields["shared_energy"]]).astype(np.float64)
        if n_links.sum() != len(idx):
            raise ValueError("%s does not add up to the length of %s" % (fields["n_links"], fields["index"]))
        seg = np.repeat(np.arange(n_nodes), n_links)
        node_e = self.n_energy[seg]
        weight = np.divide(shared, node_e, out=np.zeros_like(shared), where=node_e > 0)
        if truth["min_link_fraction"] > 0:
            keep = weight >= truth["min_link_fraction"]
            seg, idx, weight = seg[keep], idx[keep], weight[keep]
        return _segment_argbest(seg, idx, weight, n_nodes, prefer="max")

    # -------------------------------------------------------------- #
    def particle_mask(self, valid_only):
        return self.p_valid if valid_only else np.ones(len(self.p_e), dtype=bool)

    def event_slices(self, i):
        """``(particle slice, node slice)`` of event ``i`` into the flat arrays."""
        return (slice(int(self.p_off[i]), int(self.p_off[i + 1])),
                slice(int(self.n_off[i]), int(self.n_off[i + 1])))

    def node_feature(self, name):
        """Raw (untransformed) node input feature ``name`` as the dataset derives it:
        ``e`` (the node energy branch), ``eta``, ``phi``, ``cosphi``, ``sinphi``, ``r`` (the
        3D radius, HGCDataset's topo_R), ``x``, ``y``, ``z``.  Computed on demand."""
        if name == "e":
            return self.n_energy
        if name in ("eta", "phi", "x", "y", "z"):
            return getattr(self, "n_%s" % name)
        if name == "cosphi":
            return np.cos(self.n_phi)
        if name == "sinphi":
            return np.sin(self.n_phi)
        if name == "r":
            return np.sqrt(self.n_x ** 2 + self.n_y ** 2 + self.n_z ** 2)
        raise KeyError("unknown node feature %r (known: e, eta, phi, cosphi, sinphi, r, x, y, z)" % name)

    @property
    def has_boundary_truth(self):
        """Whether the file carries the entry point of the true particles (boundary x and y)."""
        return self.p_boundary_x is not None and self.p_boundary_y is not None

    @property
    def has_boundary_pred(self):
        """Whether the model regressed the entry point (boundary_x and boundary_y)."""
        return self.p_pred_boundary_x is not None and self.p_pred_boundary_y is not None

    def pred_xyz(self):
        """Cartesian position of the predicted clusters from the regressed (r, eta, phi),
        ``r`` being the 3D radius; ``None`` when one of them was not regressed."""
        r, eta, phi = self.p_pred_r, self.p_pred_eta, self.p_pred_phi
        if r is None or eta is None or phi is None:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            rt = r / np.cosh(eta)
            return rt * np.cos(phi), rt * np.sin(phi), r * np.tanh(eta)

    def summary(self):
        return dict(
            file=os.path.basename(self.path),
            topocluster_collection=self.topocluster_collection,
            particle_collection=self.particle_collection,
            node_label=self.node_label,
            particle_label=self.particle_label,
            energy_label=self.energy_label,
            truth_mode=self.truth_mode,
            regressed_fields=self.regressed_fields,
            boundary_truth=bool(self.has_boundary_truth),
            boundary_regressed=bool(self.has_boundary_pred),
            n_events=self.n_events,
            n_particles=len(self.p_e),
            n_nodes=len(self.n_energy),
            valid_fraction=float(self.p_valid.mean()) if len(self.p_valid) else float("nan"),
        )


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #

def node_edges(nmax):
    """Node-multiplicity binning that always reaches ``nmax``.

    A LayerCluster shower has O(100) nodes, a RecHit one O(1000), so the upper
    edge cannot be hard-coded.
    """
    e = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    while e[-1] < max(int(nmax), 1):
        e.append(e[-1] * 2)
    return np.array(e, dtype=float)


def robust_sigma(x):
    """IQR/1.349 - insensitive to the long tails of a resolution distribution."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    q16, q84 = np.percentile(x, [15.865, 84.135])
    return 0.5 * (q84 - q16)


def stat_label(x, unit=""):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "empty"
    return (r"$\mu_{1/2}$=%.3g%s  $\sigma_{68}$=%.3g%s  N=%d"
            % (np.median(x), unit, robust_sigma(x), unit, len(x)))


def binomial_err(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    err = np.sqrt(np.clip(p * (1 - p), 0, None) / np.where(n > 0, n, np.nan))
    return p, err


def profile(x, y, edges, stat="median"):
    """Return bin centres, the chosen statistic of y per bin, its spread and N."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    which = np.digitize(x, edges) - 1
    ctr = 0.5 * (edges[:-1] + edges[1:])
    val = np.full(len(ctr), np.nan)
    spread = np.full(len(ctr), np.nan)
    n = np.zeros(len(ctr), dtype=int)
    for i in range(len(ctr)):
        sel = which == i
        n[i] = sel.sum()
        if n[i] < 3:
            continue
        yy = y[sel]
        val[i] = np.median(yy) if stat == "median" else yy.mean()
        spread[i] = robust_sigma(yy) if stat == "median" else yy.std() / np.sqrt(n[i])
    return ctr, val, spread, n


def eff_profile(x, ok, edges):
    """Binomial efficiency of boolean ``ok`` in bins of ``x``."""
    x = np.asarray(x, float)
    ok = np.asarray(ok, bool)
    good = np.isfinite(x)
    x, ok = x[good], ok[good]
    which = np.digitize(x, edges) - 1
    ctr = 0.5 * (edges[:-1] + edges[1:])
    k = np.zeros(len(ctr)); n = np.zeros(len(ctr))
    for i in range(len(ctr)):
        sel = which == i
        n[i] = sel.sum()
        k[i] = ok[sel].sum()
    p, e = binomial_err(k, n)
    return ctr, p, e, n.astype(int)


# --------------------------------------------------------------------------- #
# scale dicts: numpy port of hepattn.utils.scaling (the chain must not import torch)
# --------------------------------------------------------------------------- #

#: node input feature -> key of the node scale dict, as HGCDataset.load_event applies it:
#: e, eta and r are scaled under their own name, the time under ``t``; phi / cosphi / sinphi
#: are fed untransformed.
NODE_SCALE_KEY = {"e": "e", "eta": "eta", "r": "r", "time": "t"}

#: plot label, unit and whether a log axis suits it, per raw node feature
NODE_FEATURE_INFO = {
    "e": dict(label=r"$E$", unit="GeV", log=True),
    "eta": dict(label=r"$\eta$", unit="", log=False),
    "phi": dict(label=r"$\phi$", unit="rad", log=False),
    "cosphi": dict(label=r"$\cos\phi$", unit="", log=False),
    "sinphi": dict(label=r"$\sin\phi$", unit="", log=False),
    "r": dict(label=r"$r$", unit="cm", log=False),
    "x": dict(label=r"$x$", unit="cm", log=False),
    "y": dict(label=r"$y$", unit="cm", log=False),
    "z": dict(label=r"$z$", unit="cm", log=False),
    "time": dict(label=r"$t$", unit="ns", log=False),
}


class VarTransformNP:
    """One entry of a scale dict, with the maths of ``hepattn.utils.scaling.VarTransform``:
    ``fn`` (``log``, ``log1p`` or ``sqrt``) first, then ``(x - shift) / scale`` where
    ``std`` -> (mean, std), ``min_max`` -> (min, max - min) and ``min_max_sym`` ->
    ((max + min)/2, (max - min)/2).  An unknown ``fn`` is ignored, as hepattn does."""

    FNS = ("log", "log1p", "sqrt")

    def __init__(self, name, config=None):
        self.name = name
        self.config = dict(config or {})
        self.type = self.config.get("type", "std")
        self.fn = self.config.get("fn", None)
        if self.type == "std":
            self.shift = float(self.config.get("mean", 0.0))
            self.scale = float(self.config.get("std", 1.0))
        elif self.type == "min_max":
            self.shift = float(self.config.get("min", 0.0))
            self.scale = float(self.config.get("max", 1.0)) - self.shift
        elif self.type == "min_max_sym":
            lo, hi = float(self.config.get("min", 0.0)), float(self.config.get("max", 1.0))
            self.shift, self.scale = (hi + lo) / 2, (hi - lo) / 2
        else:
            raise ValueError("unknown scaling type %r for %r" % (self.type, name))
        if self.fn is not None and self.fn not in self.FNS:
            print("[scale] WARNING: unknown fn %r of %r is ignored (hepattn ignores it too)" % (self.fn, name))

    @property
    def is_identity(self):
        return self.fn not in self.FNS and self.shift == 0.0 and self.scale == 1.0

    def describe(self):
        inner = "%s(x)" % self.fn if self.fn in self.FNS else "x"
        return "(%s - %.4g) / %.4g" % (inner, self.shift, self.scale)

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.fn == "log":
                x = np.log(x)
            elif self.fn == "log1p":
                x = np.log1p(x)
            elif self.fn == "sqrt":
                x = np.sqrt(x)
            return (x - self.shift) / self.scale

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float64) * self.scale + self.shift
        with np.errstate(over="ignore"):
            if self.fn == "log":
                return np.exp(x)
            if self.fn == "log1p":
                return np.expm1(x)
            if self.fn == "sqrt":
                return np.square(x)
        return x


def load_scale_dict(path):
    """``{variable: VarTransformNP}`` from a scale-dict yaml (compute_norm_par.py output)."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - both project environments ship pyyaml
        raise ImportError("reading the scale dict %s needs pyyaml" % path) from exc
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("%s is not a mapping of variable -> transform" % path)
    return {str(name): VarTransformNP(str(name), cfg) for name, cfg in raw.items()}


class FeatureScalerNP:
    """A scale dict as a lookup that, like hepattn's non-strict ``FeatureScaler`` (the one
    ``make_feature_scaler`` falls back to), returns the identity for a missing variable,
    with a warning printed once per variable."""

    def __init__(self, path_or_dict):
        if isinstance(path_or_dict, dict):
            self.path = None
            self.transforms = {k: (v if isinstance(v, VarTransformNP) else VarTransformNP(k, v))
                               for k, v in path_or_dict.items()}
        else:
            self.path = str(path_or_dict)
            self.transforms = load_scale_dict(self.path)
        self._warned = set()

    def keys(self):
        return self.transforms.keys()

    def __contains__(self, name):
        return name in self.transforms

    def __getitem__(self, name):
        tr = self.transforms.get(name)
        if tr is None:
            if name not in self._warned:
                print("[scale] WARNING: %r is not in the scale dict%s: identity transform"
                      % (name, (" " + self.path) if self.path else ""))
                self._warned.add(name)
            tr = VarTransformNP(name, {"type": "std"})
        return tr


#: also write a PDF next to every PNG (set to False for quick runs / tests)
SAVE_PDF = True


def savefig(fig, outdir, name, log_pdf=True):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, name + ".png")
    fig.savefig(png, bbox_inches="tight")
    if log_pdf and SAVE_PDF:
        fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return png
