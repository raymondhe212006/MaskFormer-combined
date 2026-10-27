"""Tests for the evaluation plotting chain (src/scripts/plot) on synthetic prediction files.

Builds tiny parquet files with the schema HGCPredictionWriter produces for every supported
(topocluster, particle) pair and checks that the collections are detected, the truth
association is reduced the right way and the whole chain runs without a failing section.
Needs only awkward, numpy, pyarrow, matplotlib (mplhep optional):

    PYTHONPATH=src python -m pytest tests/test_plot_chain.py -q
"""

import json
import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "scripts" / "plot"))
sys.path.insert(0, str(REPO / "src"))

import hgcal_eval_common as hec  # noqa: E402
from hgcal_eval_common import EvalData, detect_pair  # noqa: E402
from run_plot_chain import run_chain  # noqa: E402

hec.SAVE_PDF = False  # PNGs are enough to exercise the code and are 10x faster

PDGS = np.array([22, 22, 11, -11, 211, -211, 2112, 130, 13, 22])

#: a node scale dict in the compute_norm_par.py format (hepattn VarTransform entries)
SCALE_DICT_YAML = """# synthetic node scale dict
e:
  type: std
  fn: log
  mean: 1.0
  std: 1.5
eta:
  type: min_max_sym
  min: 1.4
  max: 3.1
r:
  type: min_max_sym
  min: 320.0
  max: 400.0
t:
  type: min_max_sym
  min: 10.0
  max: 17.0
"""


@pytest.fixture
def scale_dict_file(tmp_path):
    p = tmp_path / "node_var_transform.yaml"
    p.write_text(SCALE_DICT_YAML)
    return str(p)
PAIRS = [
    ("RecHitHGC", "SimCluster"),
    ("RecHitHGC", "MergedSimCluster"),
    ("LayerCluster", "SimCluster"),
    ("ticlTrackstersCLUE3DHigh", "ticlSimTracksters"),
]


class Truth:
    """What the generator used, kept to check EvalData against."""

    def __init__(self):
        self.node_true = []      # per node: event-local particle index, -1 noise
        self.node_wrong = []     # per node: a different particle (or -1)


def _positions(rng, n, eta_range=(1.7, 2.8)):
    eta = rng.uniform(*eta_range, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    z = np.full(n, 330.0)
    rt = z / np.sinh(eta)
    return eta, phi, rt * np.cos(phi), rt * np.sin(phi), z


def build_file(path, topo, part, rng, n_events=25, with_group=True, cp_seeded=False,
               with_boundary=True, pred_boundary=True):
    """Write a synthetic prediction parquet for the pair; returns the generator truth.

    ``cp_seeded`` (ticlSimTracksters only) makes particle 0 of every event a photon that
    entered HGCAL intact: seeded by its CaloParticle, not by a SimCluster.
    ``with_boundary`` (ticlSimTracksters only) adds the boundary branches (the entry point
    into HGCAL, with the production's broken boundaryPhi = +-eta) and ``pred_boundary`` the
    model's boundary_x / boundary_y predictions (true value + N(0, 2 cm)).
    """
    truth = Truth()
    truth.boundary_x, truth.boundary_y = [], []
    P, N, S, A = {}, {}, {}, {}   # particle, node, seed, association (per-event lists)
    top = dict(event_number=[], n_fake_particles=[])

    def add(d, key, val):
        d.setdefault(key, []).append(np.asarray(val))

    for ev in range(n_events):
        n_p = int(rng.integers(2, 6))
        pdg = PDGS[rng.integers(0, len(PDGS), n_p)]
        if cp_seeded and part == "ticlSimTracksters":
            pdg[0] = 22
        e = np.exp(rng.uniform(np.log(0.5), np.log(60.0), n_p))
        eta, phi, x, y, z = _positions(rng, n_p)
        r = np.sqrt(x**2 + y**2 + z**2)
        # nodes: a few per particle + noise
        n_per = rng.integers(3, 9, n_p)
        node_true = np.repeat(np.arange(n_p), n_per)
        n_noise = int(rng.integers(1, 4))
        node_true = np.concatenate([node_true, -np.ones(n_noise, int)])
        perm = rng.permutation(len(node_true))
        node_true = node_true[perm]
        n_n = len(node_true)
        frac = rng.dirichlet(np.ones(n_n))
        node_e = np.where(node_true >= 0, e[np.maximum(node_true, 0)] * frac * 3, rng.uniform(0.01, 0.2, n_n))
        n_eta = np.where(node_true >= 0, eta[np.maximum(node_true, 0)], rng.uniform(1.7, 2.8, n_n)) + rng.normal(0, 0.02, n_n)
        n_phi = np.where(node_true >= 0, phi[np.maximum(node_true, 0)], rng.uniform(-np.pi, np.pi, n_n)) + rng.normal(0, 0.02, n_n)
        n_z = 320.0 + rng.uniform(0, 30, n_n)
        n_rt = n_z / np.sinh(n_eta)
        n_x, n_y = n_rt * np.cos(n_phi), n_rt * np.sin(n_phi)
        # a wrong candidate per node (another particle), -1 if none exists
        wrong = (node_true + 1) % n_p if n_p > 1 else -np.ones(n_n, int)
        wrong = np.where(node_true >= 0, wrong, -1)
        truth.node_true.append(node_true)
        truth.node_wrong.append(wrong)

        # model output: mostly right, some scrambled
        pred_local = node_true.copy()
        flip = rng.random(n_n) < 0.15
        pred_local[flip] = rng.integers(-2, n_p, flip.sum())
        pred_e = e * np.exp(rng.normal(-0.1, 0.25, n_p))
        pred_eta = eta + rng.normal(0, 0.02, n_p)
        pred_phi = phi + rng.normal(0, 0.02, n_p)
        pred_r = r + rng.normal(0, 3.0, n_p)
        pred_class = hec.pdg_to_class(pdg).astype(np.int64)
        pred_class[rng.random(n_p) < 0.1] = hec.RESIDUAL_CLASS
        pred_valid = rng.random(n_p) < 0.9
        pred_n_nodes = np.bincount(pred_local[pred_local >= 0], minlength=n_p)

        # ----- particle collection -----
        if part == "ticlSimTracksters":
            n_seed = n_p + int(rng.integers(0, 3))
            extra_pdg = PDGS[rng.integers(0, len(PDGS), n_seed - n_p)]
            if cp_seeded:
                extra_pdg[:] = 2112
            seed_pdg = np.concatenate([pdg, extra_pdg])
            seed_perm = rng.permutation(n_seed)           # SimCluster order
            seed_pdg = seed_pdg[seed_perm]
            seed_row = np.argsort(seed_perm)              # particle i / extra seed -> its seed row
            seed_index = seed_row[:n_p].copy()
            assert np.all(seed_pdg[seed_index] == pdg)
            rec_energy = rng.uniform(0.5, 60, n_seed)
            group = rng.integers(0, max(n_seed // 2, 1), n_seed)
            seed_energy = rng.uniform(0.5, 60, n_seed)
            is_cp = np.zeros(n_p, bool)
            if cp_seeded:
                # Particle 0 entered HGCAL intact: seeded by its CaloParticle, whose row
                # (100 + ev) is no SimCluster row. That CaloParticle's SimClusters are the
                # particle's own one (the most energetic) and the extra ones (neutrons, so
                # picking the wrong one would change the class).
                cp_row = 100 + ev
                is_cp[0] = True
                seed_index[0] = cp_row
                group[seed_row[0]] = cp_row
                seed_energy[seed_row[0]] = 100.0
                group[seed_row[n_p:]] = cp_row
                seed_energy[seed_row[n_p:]] = 0.1
            add(S, "SimCluster_pdgId", seed_pdg)
            add(S, "SimCluster_recEnergy", rec_energy)
            add(S, "SimCluster_boundaryEnergy", seed_energy)
            if with_group:
                add(S, "SimCluster_CaloPartIdx", group)
            add(P, "ticlSimTracksters_seedIndex", seed_index)
            add(P, "ticlSimTracksters_seedIsCaloParticle", is_cp)
            add(P, "ticlSimTracksters_regressed_energy", e)
            add(P, "ticlSimTracksters_raw_energy", e * rng.uniform(0.7, 1.0, n_p))
            for k, v in dict(eta=eta, phi=phi, x=x, y=y, z=z).items():
                add(P, "ticlSimTracksters_barycenter_%s" % k, v)
            if with_boundary:
                # entry point: the position projected onto the front face (|z| = 321 cm)
                # along the line from the origin, and the momentum there (massless)
                zb = np.full(n_p, 321.05)
                bx, by = x * zb / z, y * zb / z
                beta = np.arcsinh(zb / np.hypot(bx, by))
                add(P, "ticlSimTracksters_boundaryX", bx)
                add(P, "ticlSimTracksters_boundaryY", by)
                add(P, "ticlSimTracksters_boundaryZ", zb)
                add(P, "ticlSimTracksters_boundaryEta", beta)
                # the CMSSW_20_0_0_pre1 production dumps +-eta here: nothing may rely on it
                add(P, "ticlSimTracksters_boundaryPhi", np.where(rng.random(n_p) < 0.5, beta, -beta))
                pt = e / np.cosh(eta)
                add(P, "ticlSimTracksters_boundaryPx", pt * np.cos(phi))
                add(P, "ticlSimTracksters_boundaryPy", pt * np.sin(phi))
                add(P, "ticlSimTracksters_boundaryPz", pt * np.sinh(eta))
                truth.boundary_x.append(bx)
                truth.boundary_y.append(by)
                if pred_boundary:
                    add(P, "ticlSimTracksters_pred_boundary_x", (bx + rng.normal(0, 2.0, n_p)).astype(np.float32))
                    add(P, "ticlSimTracksters_pred_boundary_y", (by + rng.normal(0, 2.0, n_p)).astype(np.float32))
        else:
            add(P, "%s_pdgId" % part, pdg)
            add(P, "%s_recEnergy" % part, e)
            add(P, "%s_boundaryEnergy" % part, e * rng.uniform(1.0, 1.2, n_p))
            for k, v in dict(eta=eta, phi=phi, x=x, y=y, z=z).items():
                add(P, "%s_impactPoint_%s" % (part, k), v)
            if with_group and part == "SimCluster":
                add(P, "SimCluster_CaloPartIdx", rng.integers(0, max(n_p // 2, 1), n_p))
        for k, v in dict(e=pred_e, eta=pred_eta, sinphi=np.sin(pred_phi), cosphi=np.cos(pred_phi),
                         r=pred_r, phi=pred_phi).items():
            add(P, "%s_pred_%s" % (part, k), v.astype(np.float32))
        add(P, "%s_pred_class" % part, pred_class)
        add(P, "%s_pred_valid" % part, pred_valid)
        add(P, "%s_pred_n_nodes" % part, pred_n_nodes.astype(np.int64))

        # ----- node collection + truth -----
        if topo == "RecHitHGC":
            add(N, "RecHitHGC_energy", node_e)
            for k, v in dict(x=n_x, y=n_y, z=n_z).items():
                add(N, "RecHitHGC_%s" % k, v)
            add(N, "RecHitHGC_%sBestMatchIdx" % part, node_true)
            add(N, "RecHitHGC_%sBestMatchQual" % part, np.where(node_true >= 0, 1.0, 0.0))
        elif topo == "LayerCluster":
            add(N, "LayerCluster_energy", node_e)
            for k, v in dict(x=n_x, y=n_y, z=n_z, eta=n_eta, phi=n_phi).items():
                add(N, "LayerCluster_%s" % k, v)
            # candidates: the true one with a SMALL score, the wrong one with a LARGE score
            nmatch, idx, qual = [], [], []
            for t, w in zip(node_true, wrong):
                cands = [] if t < 0 else ([(t, 0.05)] + ([(w, 0.8)] if w >= 0 else []))
                if t >= 0 and w >= 0 and rng.random() < 0.5:
                    cands = cands[::-1]      # order must not matter
                nmatch.append(len(cands))
                idx += [c[0] for c in cands]
                qual += [c[1] for c in cands]
            add(N, "LayerCluster_SimClusterNumMatch", np.array(nmatch, np.int64))
            add(N, "LayerCluster_SimCluster_MatchIdx", np.array(idx, np.int64))
            add(N, "LayerCluster_SimCluster_MatchQual", np.array(qual, np.float64))
        else:
            add(N, "ticlTrackstersCLUE3DHigh_raw_energy", node_e)
            for k, v in dict(x=n_x, y=n_y, z=n_z, eta=n_eta, phi=n_phi).items():
                add(N, "ticlTrackstersCLUE3DHigh_barycenter_%s" % k, v)
            # trackster time, with the -99 "no valid time" sentinel of the real files
            add(N, "ticlTrackstersCLUE3DHigh_time",
                np.where(rng.random(n_n) < 0.3, -99.0, rng.uniform(11.0, 16.0, n_n)))
            # links: the true particle owns 90% of the node energy, the wrong one 5%
            n_links, idx, shared, score = [], [], [], []
            for t, w, en in zip(node_true, wrong, node_e):
                links = [] if t < 0 else ([(t, 0.9 * en, 0.1)] + ([(w, 0.05 * en, 0.9)] if w >= 0 else []))
                if len(links) == 2 and rng.random() < 0.5:
                    links = links[::-1]
                n_links.append(len(links))
                idx += [l[0] for l in links]
                shared += [l[1] for l in links]
                score += [l[2] for l in links]
            T = "CLUE3DHighToSimTSByHits"
            add(A, "%s_n%sLinks" % (T, T), np.array(n_links, np.int64))
            add(A, "%sLinks_index" % T, np.array(idx, np.int64))
            add(A, "%sLinks_sharedEnergy" % T, np.array(shared, np.float64))
            add(A, "%sLinks_score" % T, np.array(score, np.float64))
        add(N, "%s_pred_match_idx" % topo, pred_local.astype(np.int64))
        top["event_number"].append(ev)
        top["n_fake_particles"].append(int(rng.integers(0, 3)))

    def record(d):
        return ak.zip({k: ak.Array(v) for k, v in d.items()}, depth_limit=1)

    out = {topo: record(N), part: record(P)}
    if S:
        out["SimCluster"] = record(S)
    if A:
        out["CLUE3DHighToSimTSByHits"] = record(A)
    out["event_number"] = ak.Array(np.array(top["event_number"], np.int64))
    out["n_fake_particles"] = ak.Array(np.array(top["n_fake_particles"], np.int64))
    # snappy: available in every pyarrow build (the LCG one lacks awkward's default zstd)
    ak.to_parquet(ak.zip(out, depth_limit=1), path, compression="snappy")
    truth.node_true = np.concatenate(truth.node_true)
    truth.node_wrong = np.concatenate(truth.node_wrong)
    if truth.boundary_x:
        truth.boundary_x = np.concatenate(truth.boundary_x)
        truth.boundary_y = np.concatenate(truth.boundary_y)
    return truth


@pytest.fixture(params=PAIRS, ids=["_".join(p) for p in PAIRS])
def synthetic(request, tmp_path):
    topo, part = request.param
    path = tmp_path / "pred__test.parquet"
    truth = build_file(str(path), topo, part, np.random.default_rng(1234))
    return topo, part, str(path), truth


def test_detect_and_truth(synthetic):
    topo, part, path, truth = synthetic
    assert detect_pair(path) == (topo, part)
    with pytest.raises(ValueError):
        detect_pair(path, topocluster_collection="NoSuchCollection")

    data = EvalData(path)
    assert data.topocluster_collection == topo and data.particle_collection == part
    assert len(data.n_true_local) == len(truth.node_true)
    # the truth association must be recovered for every pair: best-match index, the
    # candidate with the SMALLEST score, or the link with the largest energy share
    np.testing.assert_array_equal(data.n_true_local, truth.node_true)
    wrong_possible = truth.node_wrong >= 0
    assert not np.any(data.n_true_local[wrong_possible] == truth.node_wrong[wrong_possible])
    assert data.p_class.min() >= 0 and set(np.unique(data.p_class)) <= set(hec.PHYSICS_CLASSES)
    assert data.p_pred_phi is not None and data.p_pred_r is not None
    assert (data.p_group is not None) == (part in ("SimCluster", "ticlSimTracksters"))
    if part == "ticlSimTracksters":
        assert data.truth_mode == "links" and data.energy_label == "regressed_energy"
        assert data.energy_alt_label == "raw_energy" and data.p_e_alt is not None
        # the entry point and its prediction; eta/phi derived from the position, never
        # from the dumped (broken) boundaryPhi
        assert data.has_boundary_truth and data.has_boundary_pred
        np.testing.assert_allclose(data.p_boundary_x, truth.boundary_x, rtol=1e-6)
        np.testing.assert_allclose(data.p_boundary_y, truth.boundary_y, rtol=1e-6)
        np.testing.assert_allclose(data.p_boundary_phi, np.arctan2(truth.boundary_y, truth.boundary_x), atol=1e-6)
        np.testing.assert_allclose(data.p_boundary_eta, np.arcsinh(321.05 / np.hypot(truth.boundary_x, truth.boundary_y)), atol=1e-5)
        assert np.all(np.abs(data.p_pred_boundary_x - data.p_boundary_x) < 15.0)
        assert "boundary_x" in data.regressed_fields and data.p_pred_boundary_z is None
    else:
        assert not data.has_boundary_pred
        # the SimCluster impact point is the entry point
        assert data.has_boundary_truth == (part in ("SimCluster", "MergedSimCluster"))
        if data.has_boundary_truth:
            np.testing.assert_allclose(data.p_boundary_x, data.p_x)
    if part == "ticlSimTracksters":
        pass
    elif topo == "LayerCluster":
        assert data.truth_mode == "ragged"
    else:
        assert data.truth_mode == "best" and data.energy_label == "recEnergy"


def test_full_chain(synthetic, tmp_path, scale_dict_file):
    topo, part, path, truth = synthetic
    outdir = tmp_path / "plots"
    node_features = ["e", "eta", "cosphi", "sinphi", "r", "time"]
    result = run_chain(path, str(outdir), scale_dict=scale_dict_file,
                       node_features=node_features, n_display=2)
    assert result["failures"] == {}, "\n".join(result["failures"].values())
    boundary = part == "ticlSimTracksters"     # the synthetic trackster file regresses the entry point
    feats = ["e", "eta", "sinphi", "cosphi", "r"] + (["boundary_x", "boundary_y"] if boundary else [])
    ratio_feats = [k for k in feats if not k.startswith("boundary")]   # signed x/y: no pred/true
    subs = ["resolution", "resolution_by_class", "efficiency", "diagnostics", "energy",
            "cluster_properties", "cluster_properties/response", "topocluster_features/raw",
            "topocluster_features/transform", "event_reconstruction", "event_display"]
    subs += ["cluster_properties/response/by_%s" % k for k in feats]
    if boundary:
        subs.append("boundary_position")
    for sub in subs:
        assert any(f.suffix == ".png" for f in (outdir / sub).iterdir()), sub

    def pngs(sub):
        return sorted(f.name for f in (outdir / sub).glob("*.png"))

    assert pngs("cluster_properties") == sorted(["dist_%s.png" % k for k in feats] + ["dist_e_logy.png"])
    assert pngs("cluster_properties/response") == sorted("response_%s.png" % k for k in ratio_feats)
    for x in feats:
        assert pngs("cluster_properties/response/by_%s" % x) == sorted(
            "response_%s_vs_%s.png" % (k, x) for k in ratio_feats)
    for k in feats:
        assert "resolution_%s.png" % k in pngs("resolution")
        assert k == "e" or "sigma68_%s_vs_energy.png" % k in pngs("resolution")
    # only the synthetic trackster file carries a time branch; elsewhere "time" is skipped
    node_present = node_features if topo == "ticlTrackstersCLUE3DHigh" else node_features[:-1]
    assert pngs("topocluster_features/raw") == sorted("node_%s.png" % k for k in node_present)
    assert pngs("topocluster_features/transform") == sorted("node_%s_transformed.png" % k for k in node_present)
    disp = pngs("event_display")
    assert len(disp) == 6 and all(n.startswith("event_") for n in disp)
    for name in ("total_energy_response.png", "weighted_eta_residual.png", "n_fake_particles.png"):
        assert name in pngs("event_reconstruction"), name
    metrics = json.loads((outdir / "metrics.json").read_text())
    assert metrics["dataset"]["particle_collection"] == part
    assert "clustering_efficiency" in metrics and "classification" in metrics
    assert ("sample_structure" in metrics) == (part in ("SimCluster", "ticlSimTracksters"))
    assert (outdir / "summary.txt").exists()
    assert (outdir / "energy" / "energy_metrics.json").exists()
    assert (outdir / "energy" / "energy_summary.txt").exists()
    assert not (outdir / "plot_chain_failures.txt").exists()
    # the summary names the actual collections
    txt = (outdir / "summary.txt").read_text()
    assert part in txt and topo in txt
    assert "cluster response" in txt and "event reconstruction" in txt
    # build_file smears E by exp(N(-0.1, 0.25)) and eta by N(0, 0.02)
    resp = metrics["cluster_properties"]["response"]
    assert abs(resp["e"]["all"]["median"] - np.exp(-0.1)) < 0.1
    assert abs(resp["eta"]["all"]["median"] - 1.0) < 0.02
    assert set(metrics["cluster_properties"]["response_vs"]["e"]) == set(feats)
    tr = metrics["topocluster_features"]["transform"]
    assert tr["e"]["key"] == "e" and tr["e"]["fn"] == "log"
    assert tr["cosphi"]["key"] is None           # fed untransformed, as the dataset does
    assert tr["eta"]["frac_outside"] < 0.01      # synthetic eta in [1.7, 2.8], dict range [1.4, 3.1]
    if topo == "ticlTrackstersCLUE3DHigh":
        assert metrics["topocluster_features"]["raw"]["time"]["n_sentinel"] > 0
        assert tr["time"]["key"] == "t"
    er = metrics["event_reconstruction"]
    assert er["n_events"] == metrics["dataset"]["n_events"]
    assert abs(er["total_energy_response"]["all"]["median"] - np.exp(-0.1)) < 0.15
    assert abs(er["weighted_residual"]["eta"]["all"]["median"]) < 0.02
    assert "n_fake_particles" in er
    assert metrics["dataset"]["boundary_regressed"] == boundary
    assert ("boundary_position" in metrics) == boundary
    assert ("weighted_boundary_x_residual.png" in pngs("event_reconstruction")) == boundary
    if boundary:
        bp = metrics["boundary_position"]
        expected = sorted(["xy_pred_vs_true.png", "residual_xy_2d.png", "distance.png", "distance_logx.png",
                           "distance_vs_energy.png", "distance_vs_eta.png", "distance_by_class.png",
                           "residual_vs_true_x.png", "residual_vs_true_y.png", "entry_point_map.png"])
        assert pngs("boundary_position") == expected
        # build_file smears the entry point by N(0, 2 cm) per coordinate: the distance is
        # Rayleigh with median 2 sqrt(2 ln 2) = 2.35 cm, the 1D residuals have sigma68 ~ 2 cm
        assert abs(bp["distance"]["model_all"]["median"] - 2.0 * np.sqrt(2 * np.log(2))) < 0.5
        assert abs(bp["residual"]["x"]["sigma68"] - 2.0) < 0.5 and abs(bp["residual"]["y"]["median"]) < 0.5
        assert set(bp["distance"]) == {"model_all", "model_valid", "baseline_pred_nodes", "baseline_true_nodes"}
        # the synthetic nodes sit at z = 320-350 cm around the true direction: the projected
        # barycenter baseline lands within a few cm too
        assert bp["distance"]["baseline_true_nodes"]["median"] < 15.0
        assert set(bp["by_class"]) <= {hec.class_name(c) for c in hec.PHYSICS_CLASSES}
        assert "boundary_x_all" in metrics["resolution_inclusive"]
        assert abs(metrics["resolution_inclusive"]["boundary_x_all"]["sigma68"] - 2.0) < 0.5
        assert "entrance position" in txt


def test_links_min_fraction_and_seed_pdg(tmp_path):
    path = tmp_path / "ts__test.parquet"
    truth = build_file(str(path), "ticlTrackstersCLUE3DHigh", "ticlSimTracksters", np.random.default_rng(7))
    # every link carries <= 90% of the node energy: a 95% threshold drops them all
    data = EvalData(str(path), min_link_fraction=0.95)
    assert np.all(data.n_true_local == -1)
    # the 50% threshold keeps only the true link
    data = EvalData(str(path), min_link_fraction=0.5)
    np.testing.assert_array_equal(data.n_true_local, truth.node_true)
    assert np.all(np.isfinite(data.n_true_qual[data.n_true_local >= 0]))
    # an unknown association table is refused
    with pytest.raises(ValueError):
        EvalData(str(path), association_collection="CLUE3DHighToSimTSByLCs")


def test_no_group_branch(tmp_path):
    path = tmp_path / "sc__test.parquet"
    build_file(str(path), "RecHitHGC", "SimCluster", np.random.default_rng(3), n_events=10, with_group=False)
    data = EvalData(str(path))
    assert data.p_group is None
    result = run_chain(str(path), str(tmp_path / "plots"), skip_energy=True)
    assert result["failures"] == {}
    assert "sample_structure" not in result["metrics"]
    assert result["energy_metrics"] is None
    # no scale dict: the raw node features only, no transform/ folder
    assert (tmp_path / "plots" / "topocluster_features" / "raw").is_dir()
    assert not (tmp_path / "plots" / "topocluster_features" / "transform").exists()
    assert "transform" not in result["metrics"]["topocluster_features"]
    assert (tmp_path / "plots" / "event_display").is_dir()


def test_caloparticle_seeded_particles(tmp_path):
    """SimTracksters seeded by the CaloParticle itself (unconverted photons): seedIndex is a
    CaloParticle row, resolved through SimCluster_CaloPartIdx to its most energetic SimCluster."""
    n_events = 12
    path = tmp_path / "cp__test.parquet"
    truth = build_file(str(path), "ticlTrackstersCLUE3DHigh", "ticlSimTracksters",
                       np.random.default_rng(11), n_events=n_events, cp_seeded=True)
    data = EvalData(str(path))
    np.testing.assert_array_equal(data.n_true_local, truth.node_true)
    first = data.p_off[:-1]                   # particle 0 of every event: the photon
    others = np.setdiff1d(np.arange(len(data.p_pdg)), first)
    assert np.all(data.p_pdg[first] == 22)    # not the 2112 of the CaloParticle's other SimClusters
    assert np.all(data.p_class[first] == hec.pdg_to_class(np.array([22]))[0])
    # the photon is a CaloParticle group of its own, distinct in every event
    assert data.p_group is not None
    assert len(np.unique(data.p_group[first])) == n_events
    assert not np.any(np.isin(data.p_group[others], data.p_group[first]))
    result = run_chain(str(path), str(tmp_path / "plots"))
    assert result["failures"] == {}, "\n".join(result["failures"].values())
    # without SimCluster_CaloPartIdx they cannot be resolved: unmapped class, but no crash
    path = tmp_path / "cp_nogroup__test.parquet"
    build_file(str(path), "ticlTrackstersCLUE3DHigh", "ticlSimTracksters",
               np.random.default_rng(11), n_events=n_events, with_group=False, cp_seeded=True)
    data = EvalData(str(path))
    first = data.p_off[:-1]
    others = np.setdiff1d(np.arange(len(data.p_pdg)), first)
    assert np.all(data.p_pdg[first] == 0) and np.all(data.p_pdg[others] != 0)
    assert data.p_group is None


def test_boundary_truth_without_prediction(tmp_path):
    """A model that did not regress the entry point, on a file that carries it (the current
    trackster models): the truth is read, nothing boundary-specific is plotted, no failure."""
    path = tmp_path / "nopred__test.parquet"
    truth = build_file(str(path), "ticlTrackstersCLUE3DHigh", "ticlSimTracksters",
                       np.random.default_rng(5), n_events=12, pred_boundary=False)
    data = EvalData(str(path))
    assert data.has_boundary_truth and not data.has_boundary_pred
    assert "boundary_x" not in data.regressed_fields
    np.testing.assert_allclose(data.p_boundary_x, truth.boundary_x, rtol=1e-6)
    result = run_chain(str(path), str(tmp_path / "plots"), skip_energy=True, skip_display=True)
    assert result["failures"] == {}, "\n".join(result["failures"].values())
    assert not (tmp_path / "plots" / "boundary_position").exists()
    assert "boundary_position" not in result["metrics"]
    assert result["metrics"]["dataset"]["boundary_truth"] is True
    assert result["metrics"]["dataset"]["boundary_regressed"] is False
    assert not any("boundary" in f.name for f in (tmp_path / "plots" / "resolution").glob("*.png"))


def test_no_boundary_branches(tmp_path):
    """An older file without the boundary branches loads and plots like before."""
    path = tmp_path / "nobnd__test.parquet"
    build_file(str(path), "ticlTrackstersCLUE3DHigh", "ticlSimTracksters",
               np.random.default_rng(6), n_events=12, with_boundary=False)
    data = EvalData(str(path))
    assert not data.has_boundary_truth and not data.has_boundary_pred
    assert data.p_boundary_x is None and data.p_boundary_phi is None and data.p_boundary_sinphi is None
    result = run_chain(str(path), str(tmp_path / "plots"), skip_energy=True, skip_display=True)
    assert result["failures"] == {}, "\n".join(result["failures"].values())
    assert not (tmp_path / "plots" / "boundary_position").exists()
