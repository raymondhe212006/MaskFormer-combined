"""Tests of the entrance-position (boundary) targets of HGCDataset on a synthetic
trackster/SimTrackster parquet: the boundary branches are loaded, the targets are scaled with
their own entries of the particle dict, hepattn's regression task gets a node-level proxy per
target, and a file without the branches keeps working unless a boundary target is asked for.
Needs torch (the project apptainer image); skipped otherwise:

    PYTHONPATH=src python -m pytest tests/test_boundary_targets.py -q
"""

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("hepattn")

from dataset.dataset_maskformers import BASE_TARGETS, EXTRA_TARGETS, HGCDataset  # noqa: E402

TOPO, PART = "ticlTrackstersCLUE3DHigh", "ticlSimTracksters"
NODE_DICT = REPO / "configs" / "hgcal_trackster_clue3dhigh_var_transform.yaml"
PARTICLE_DICT = REPO / "configs" / "hgcal_simtrackster_var_transform.yaml"
INPUTS = {"topocluster": ["e", "eta", "cosphi", "sinphi", "r"]}
PDGS = np.array([22, 11, -11, 211, -211, 130, 2112])


def build_file(path, rng, n_events=8, with_boundary=True):
    """A small trackster/SimTrackster file with everything HGCDataset reads for the pair.
    Event 0 has no trackster (dropped by the dataset). Returns the generator truth."""
    T, P, S, A = {}, {}, {}, {}
    truth = dict(boundary_x=[], boundary_y=[], boundary_z=[], node_x=[], node_y=[])

    def add(d, key, val):
        d.setdefault(key, []).append(np.asarray(val))

    for ev in range(n_events):
        n_p = int(rng.integers(2, 5))
        n_n = 0 if ev == 0 else int(rng.integers(3, 9))
        e = np.exp(rng.uniform(np.log(1.0), np.log(80.0), n_p))
        eta = rng.uniform(1.7, 2.8, n_p)
        phi = rng.uniform(-np.pi, np.pi, n_p)
        z = np.full(n_p, 335.0)
        rt = z / np.sinh(eta)
        x, y = rt * np.cos(phi), rt * np.sin(phi)
        add(P, f"{PART}_regressed_energy", e)
        add(P, f"{PART}_raw_energy", e * rng.uniform(0.6, 1.0, n_p))
        for k, v in dict(eta=eta, phi=phi, x=x, y=y, z=z).items():
            add(P, f"{PART}_barycenter_{k}", v)
        add(P, f"{PART}_seedIndex", np.arange(n_p))
        add(P, f"{PART}_seedIsCaloParticle", np.zeros(n_p, bool))
        if with_boundary:
            zb = np.full(n_p, 321.05)
            bx, by = x * zb / z, y * zb / z
            beta = np.arcsinh(zb / np.hypot(bx, by))
            add(P, f"{PART}_boundaryX", bx)
            add(P, f"{PART}_boundaryY", by)
            add(P, f"{PART}_boundaryZ", zb)
            add(P, f"{PART}_boundaryEta", beta)
            add(P, f"{PART}_boundaryPhi", np.where(rng.random(n_p) < 0.5, beta, -beta))   # the production's bug
            pt = e / np.cosh(eta)
            add(P, f"{PART}_boundaryPx", pt * np.cos(phi))
            add(P, f"{PART}_boundaryPy", pt * np.sin(phi))
            add(P, f"{PART}_boundaryPz", pt * np.sinh(eta))
            if ev > 0:
                truth["boundary_x"].append(bx); truth["boundary_y"].append(by); truth["boundary_z"].append(zb)
        pdg = PDGS[rng.integers(0, len(PDGS), n_p)]
        add(S, "SimCluster_pdgId", pdg)
        add(S, "SimCluster_eventId", np.zeros(n_p, np.int64))
        add(S, "SimCluster_bunchCrossing", np.zeros(n_p, np.int64))
        add(S, "SimCluster_CaloPartIdx", np.arange(n_p))
        add(S, "SimCluster_boundaryEnergy", e)

        owner = rng.integers(0, n_p, n_n)
        n_e = e[owner] * rng.uniform(0.2, 0.8, n_n)
        n_eta = eta[owner] + rng.normal(0, 0.02, n_n)
        n_phi = phi[owner] + rng.normal(0, 0.02, n_n)
        n_z = 322.0 + rng.uniform(0, 30, n_n)
        n_rt = n_z / np.sinh(n_eta)
        n_x, n_y = n_rt * np.cos(n_phi), n_rt * np.sin(n_phi)
        add(T, f"{TOPO}_raw_energy", n_e)
        add(T, f"{TOPO}_raw_em_energy", n_e * rng.uniform(0, 1, n_n))
        add(T, f"{TOPO}_time", np.where(rng.random(n_n) < 0.3, -99.0, rng.uniform(11, 16, n_n)))
        for k, v in dict(x=n_x, y=n_y, z=n_z, eta=n_eta, phi=n_phi).items():
            add(T, f"{TOPO}_barycenter_{k}", v)
        if ev > 0:
            truth["node_x"].append(n_x); truth["node_y"].append(n_y)
        # association: each trackster owns 90% of its energy from its particle, 5% from another
        n_links, idx, shared, score = [], [], [], []
        for o, en in zip(owner, n_e):
            links = [(o, 0.9 * en, 0.1)]
            if n_p > 1:
                links.append(((o + 1) % n_p, 0.05 * en, 0.9))
            n_links.append(len(links)); idx += [l[0] for l in links]; shared += [l[1] for l in links]; score += [l[2] for l in links]
        TBL = "CLUE3DHighToSimTSByHits"
        add(A, f"{TBL}_n{TBL}Links", np.array(n_links, np.int64))
        add(A, f"{TBL}Links_index", np.array(idx, np.int64))
        add(A, f"{TBL}Links_sharedEnergy", np.array(shared, np.float64))
        add(A, f"{TBL}Links_score", np.array(score, np.float64))

    def record(d):
        return ak.zip({k: ak.Array(v) for k, v in d.items()}, depth_limit=1)

    out = {TOPO: record(T), PART: record(P), "SimCluster": record(S), "CLUE3DHighToSimTSByHits": record(A)}
    ak.to_parquet(ak.zip(out, depth_limit=1), path, compression="snappy")
    return {k: np.concatenate(v) if v else None for k, v in truth.items()}


def dataset(path, targets, num_objects=16, max_nodes=32):
    return HGCDataset(
        filepath=str(path), inputs=INPUTS, targets={"particle": targets},
        scale_dict_path=str(NODE_DICT), particle_scale_dict_path=str(PARTICLE_DICT),
        topocluster_collection=TOPO, particle_collection=PART, num_objects=num_objects, max_nodes=max_nodes,
        incidence_cutval=0.01, normalize_incidence=False,
    )


def collect(ds, key, kind="labels"):
    """Flat valid entries of one label (per valid particle) or input (per valid node)."""
    out = []
    for i in range(len(ds)):
        inputs, labels = ds[i]
        if kind == "labels":
            out.append(labels[key][labels["particle_valid"]])
        else:
            out.append(inputs[key][inputs["node_valid"]])
    return torch.cat(out).numpy()


@pytest.fixture
def synthetic(tmp_path):
    path = tmp_path / "synthetic.parquet"
    truth = build_file(path, np.random.default_rng(42))
    return path, truth


def test_boundary_targets_and_proxies(synthetic):
    path, truth = synthetic
    ds = dataset(path, list(BASE_TARGETS) + ["boundary_x", "boundary_y"])
    assert ds.extra_targets == ["boundary_x", "boundary_y"]
    assert ds.proxy_targets == {"boundary_x": "x", "boundary_y": "y"}
    assert len(ds) == 7                                   # event 0 has no trackster
    # every boundary branch is loaded (position and momentum), eta/phi from the position
    for canonical in ("boundary_x", "boundary_y", "boundary_z", "boundary_eta", "boundary_phi",
                      "boundary_px", "boundary_py", "boundary_pz"):
        assert f"particle_{canonical}" in ds.full_data_array, canonical
    bx, by, bz = (ds.full_data_array[f"particle_boundary_{c}"].numpy() for c in "xyz")
    np.testing.assert_allclose(bx, truth["boundary_x"], rtol=1e-6)
    b_eta = ds.full_data_array["particle_boundary_eta"].numpy()
    b_phi = ds.full_data_array["particle_boundary_phi"].numpy()
    np.testing.assert_allclose(b_phi, np.arctan2(by, bx), atol=1e-6)
    np.testing.assert_allclose(b_eta, np.arcsinh(bz / np.hypot(bx, by)), atol=1e-5)
    assert not np.allclose(np.abs(b_phi), b_eta, atol=1e-3)        # not the dumped +-eta
    # the targets are the scaled entry point: inverse-transform gives the branch back
    inputs, labels = ds[0]
    for key in ("particle_boundary_x", "particle_boundary_y", "particle_e", "particle_eta"):
        assert key in labels and labels[key].shape == (ds.num_objects,)
    assert "particle_boundary_z" not in labels
    for c, raw in (("x", bx), ("y", by)):
        scaled = collect(ds, f"particle_boundary_{c}")
        tr = ds.particle_scaler.transforms[f"boundary_{c}"]
        np.testing.assert_allclose(tr.inverse_transform(torch.tensor(scaled)).numpy(), raw, rtol=1e-5, atol=1e-3)
        assert np.abs(scaled).max() < 2.0                 # min_max_sym entry fitted on the real sample
        assert not np.allclose(scaled, raw)
    # the padded part of a target is NaN, the valid part finite
    assert torch.isnan(labels["particle_boundary_x"][labels["particle_valid"].sum():]).all()
    assert torch.isfinite(labels["particle_boundary_x"][labels["particle_valid"]]).all()
    # the node-level proxies hepattn indexes: the raw trackster barycenter x / y
    for key in ("node_boundary_x", "node_boundary_y"):
        assert key in inputs and inputs[key].shape == (ds.max_nodes,)
    np.testing.assert_allclose(collect(ds, "node_boundary_x", "inputs"), truth["node_x"], rtol=1e-6)
    np.testing.assert_allclose(collect(ds, "node_boundary_y", "inputs"), truth["node_y"], rtol=1e-6)
    assert "node_boundary_z" not in inputs
    # the base inputs are untouched: the 3D node radius of the synthetic tracksters
    r = collect(ds, "node_r", "inputs")
    assert r.min() > 300.0 and r.max() < 450.0


def test_other_boundary_targets(synthetic):
    path, truth = synthetic
    ds = dataset(path, list(BASE_TARGETS) + ["boundary_z", "boundary_eta", "boundary_sinphi", "boundary_cosphi"])
    inputs, labels = ds[0]
    for t in ("boundary_z", "boundary_eta", "boundary_sinphi", "boundary_cosphi"):
        assert f"particle_{t}" in labels and f"node_{t}" in inputs
    bx, by = truth["boundary_x"], truth["boundary_y"]
    phi = np.arctan2(by, bx)
    np.testing.assert_allclose(collect(ds, "particle_boundary_sinphi"), np.sin(phi), atol=1e-6)
    np.testing.assert_allclose(collect(ds, "particle_boundary_cosphi"), np.cos(phi), atol=1e-6)
    tr = ds.particle_scaler.transforms["boundary_z"]
    np.testing.assert_allclose(tr.inverse_transform(torch.tensor(collect(ds, "particle_boundary_z"))).numpy(),
                               truth["boundary_z"], rtol=1e-5, atol=1e-3)
    # proxies: the raw node eta and sin/cos phi
    np.testing.assert_allclose(collect(ds, "node_boundary_eta", "inputs"), collect(ds, "node_eta", "inputs"))
    np.testing.assert_allclose(collect(ds, "node_boundary_sinphi", "inputs"), collect(ds, "node_sinphi", "inputs"))
    np.testing.assert_allclose(collect(ds, "node_boundary_z", "inputs"),
                               np.sqrt(collect(ds, "node_r", "inputs") ** 2 - truth["node_x"] ** 2 - truth["node_y"] ** 2), rtol=1e-5)


def test_base_targets_only(synthetic):
    path, _ = synthetic
    ds = dataset(path, list(BASE_TARGETS))
    assert ds.extra_targets == [] and ds.proxy_targets == {}
    inputs, labels = ds[0]
    assert not any(k.startswith("node_boundary") for k in inputs)
    assert not any(k.startswith("particle_boundary") for k in labels)
    # the boundary branches are still loaded ("saved") for whoever needs them
    assert "particle_boundary_px" in ds.full_data_array and "particle_boundary_x" in ds.full_data_array


def test_unknown_and_unavailable_targets(synthetic, tmp_path):
    path, _ = synthetic
    with pytest.raises(ValueError, match="unknown regression targets"):
        dataset(path, list(BASE_TARGETS) + ["boundary_r"])
    # a file without the boundary branches: fine without boundary targets, refused with them
    old = tmp_path / "old.parquet"
    build_file(old, np.random.default_rng(1), with_boundary=False)
    ds = dataset(old, list(BASE_TARGETS))
    assert len(ds) == 7
    assert not any(k.startswith("particle_boundary") for k in ds.full_data_array)
    with pytest.raises(ValueError, match="has no branch 'ticlSimTracksters_boundaryX'"):
        dataset(old, list(BASE_TARGETS) + ["boundary_x", "boundary_y"])


def test_extra_targets_table():
    assert set(EXTRA_TARGETS) == {"boundary_x", "boundary_y", "boundary_z", "boundary_eta", "boundary_sinphi", "boundary_cosphi"}
    assert set(EXTRA_TARGETS.values()) <= {"x", "y", "z", "eta", "sinphi", "cosphi"}
