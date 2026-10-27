"""Tests for the NaN/inf guards (src/dataset/finite_checks.py) and their use by the model
wrapper, on synthetic tensors. Needs numpy, scipy, torch; the wrapper test also needs hepattn
and lightning (skipped otherwise):

    PYTHONPATH=src python -m pytest tests/test_finite_checks.py -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataset.finite_checks import (  # noqa: E402
    INCIDENCE_TOL,
    assert_batch_finite,
    assert_finite,
    assert_incidence_matrix,
    assert_losses_finite,
    assert_scaler_finite,
    assert_transform_domain,
)

NAN, INF = float("nan"), float("inf")


def test_assert_finite_accepts_finite_and_non_float():
    assert_finite("x", torch.tensor([0.0, -1.5, 2.0]))
    assert_finite("x", np.array([1, 2, 3]))
    assert_finite("x", torch.tensor([True, False]))
    assert_finite("x", np.array([1.0, 2.0]))


@pytest.mark.parametrize("values", [torch.tensor([1.0, NAN]), np.array([1.0, INF]), torch.tensor([[NAN, NAN], [1.0, -INF]])])
def test_assert_finite_raises_with_counts(values):
    with pytest.raises(AssertionError, match=r"branch 'e': \d+/\d+ non-finite .*\(file.parquet\)"):
        assert_finite("branch 'e'", values, context="file.parquet")


def test_assert_transform_domain():
    class T:
        def __init__(self, fn):
            self.fn = fn

    assert_transform_domain("e", torch.tensor([0.5, 3.0]), T("log"))
    assert_transform_domain("e", torch.tensor([0.0, 3.0]), T("log1p"))
    assert_transform_domain("e", torch.tensor([-5.0]), T(None))  # identity has no domain
    with pytest.raises(AssertionError, match="outside the domain > 0 of the 'log' transform"):
        assert_transform_domain("e", torch.tensor([0.0, 3.0]), T("log"))
    with pytest.raises(AssertionError, match="> -1 of the 'log1p'"):
        assert_transform_domain("e", torch.tensor([-1.0]), T("log1p"))
    with pytest.raises(AssertionError, match=">= 0 of the 'sqrt'"):
        assert_transform_domain("e", torch.tensor([-0.1]), T("sqrt"))


def test_assert_scaler_finite():
    class T:
        def __init__(self, shift, scale):
            self.shift, self.scale = shift, scale

    class S:
        def __init__(self, **transforms):
            self.transforms = transforms

    assert_scaler_finite(S(e=T(0.5, 0.8), eta=T(2.0, 0.9)), "ok.yaml")
    with pytest.raises(AssertionError, match="variable 'r' has shift=1.0 scale=0.0"):
        assert_scaler_finite(S(e=T(0.5, 0.8), r=T(1.0, 0.0)), "bad.yaml")
    with pytest.raises(AssertionError, match="variable 'e'"):
        assert_scaler_finite(S(e=T(NAN, 1.0)), "bad.yaml")


def soft_incidence():
    # 3 particles x 4 nodes of energy shares; node 3 has no link (noise), node 0 is shared
    return csr_matrix(np.array([[0.7, 1.0, 0.0, 0.0], [0.3, 0.0, 0.2, 0.0], [0.0, 0.0, 0.05, 0.0]], dtype=np.float32))


def test_incidence_matrix_soft_accepts_shares_and_float32_rounding():
    inc = soft_incidence()
    assert_incidence_matrix(inc, row_normalized=False)
    inc.data[1] = 1 + 8e-7  # sharedEnergy / raw_energy in float32 on the tau sample
    assert_incidence_matrix(inc, row_normalized=False)
    assert_incidence_matrix(inc.toarray(), row_normalized=False)
    assert_incidence_matrix(torch.tensor(inc.toarray()).unsqueeze(0).repeat(2, 1, 1), row_normalized=False)
    assert_incidence_matrix(csr_matrix((0, 0), dtype=np.float32), row_normalized=False)  # event without particles/nodes


def test_incidence_matrix_rejects_bad_entries():
    for bad, pattern in ((NAN, "non-finite"), (INF, "non-finite"), (-0.1, r"outside \[0, 1\]"), (1.5, r"outside \[0, 1\]")):
        inc = soft_incidence()
        inc.data[0] = bad
        with pytest.raises(AssertionError, match=f"incidence matrix: .*{pattern}.*event 7"):
            assert_incidence_matrix(inc, row_normalized=False, context="event 7")


def test_incidence_matrix_column_sums_soft_and_row_sums_normalised():
    inc = soft_incidence()
    inc[1, 0] = 0.3 + 2 * INCIDENCE_TOL  # node 0 now gets 1.0002 of its energy
    with pytest.raises(AssertionError, match="node column .* sums to 1.0002"):
        assert_incidence_matrix(inc, row_normalized=False)
    # one-hot assignment: row sums exceed 1, column sums do not
    onehot = csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
    assert_incidence_matrix(onehot, row_normalized=False)
    with pytest.raises(AssertionError, match="particle row .* sums to 2"):
        assert_incidence_matrix(onehot, row_normalized=True)
    assert_incidence_matrix(csr_matrix(np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])), row_normalized=True)


def make_batch(batch_size=3, num_nodes=5, num_objects=4):
    node_valid = torch.zeros(batch_size, num_nodes, dtype=torch.bool)
    node_valid[0, :3] = True
    node_valid[2, :5] = True  # event 1 has no node
    particle_valid = torch.zeros(batch_size, num_objects, dtype=torch.bool)
    particle_valid[0, :2] = True
    particle_valid[1, :1] = True
    particle_valid[2, :4] = True
    inputs = {"node_features": torch.randn(batch_size, num_nodes, 5), "node_e": torch.rand(batch_size, num_nodes), "node_valid": node_valid}
    targets = {
        "node_valid": node_valid,
        "particle_valid": particle_valid,
        "particle_class": torch.full((batch_size, num_objects), 3),  # RESIDUAL_CLASS
        "particle_incidence": torch.rand(batch_size, num_objects, num_nodes) * 0.25,
        "particle_node_valid": torch.zeros(batch_size, num_objects, num_nodes, dtype=torch.bool),
        "event_number": torch.tensor([10, 11, 12]),
    }
    for field in ("e", "eta", "r"):
        tgt = torch.full((batch_size, num_objects), NAN)  # padded targets are NaN by construction
        tgt[particle_valid] = torch.randn(int(particle_valid.sum()))
        targets[f"particle_{field}"] = tgt
    return inputs, targets


def test_batch_finite_ignores_padding():
    inputs, targets = make_batch()
    inputs["node_features"][1, :, :] = NAN  # padded nodes of an event without tracksters
    inputs["node_e"][0, 4] = INF  # padded node
    assert_batch_finite(inputs, targets, context="val batch 0")


def test_batch_finite_reports_keys_and_events():
    inputs, targets = make_batch()
    inputs["node_features"][2, 1, 3] = NAN
    targets["particle_e"][0, 1] = INF
    targets["particle_incidence"][1, 0, 0] = NAN
    with pytest.raises(AssertionError) as err:
        assert_batch_finite(inputs, targets, context="train batch 4")
    msg = str(err.value)
    assert "train batch 4" in msg
    assert "'input node_features' non-finite for 1 events (event numbers: 12)" in msg
    assert "'target particle_e' non-finite for 1 events (event numbers: 10)" in msg
    assert "'particle_incidence' non-finite for 1 events (event numbers: 11)" in msg


def test_batch_finite_incidence_range():
    inputs, targets = make_batch()
    targets["particle_incidence"][2, 3, 4] = 1.01
    with pytest.raises(AssertionError, match=r"'particle_incidence' outside \[0, 1\] for 1 events \(event numbers: 12\)"):
        assert_batch_finite(inputs, targets)


def test_losses_finite():
    losses = {"final": {"mask": {"mask_bce": torch.tensor(0.3), "mask_dice": torch.tensor(0.1)}, "regression": {"l1": torch.tensor(2.0)}}}
    assert_losses_finite(losses, stage="val")
    losses["final"]["regression"]["l1"] = torch.tensor(NAN)
    losses["layer_1"] = {"classification": {"object_ce": torch.tensor(INF)}}
    with pytest.raises(AssertionError, match=r"non-finite val loss terms: \{'final/regression/l1': nan, 'layer_1/classification/object_ce': inf\}; batch event numbers: \[3, 4\]"):
        assert_losses_finite(losses, stage="val", event_numbers=torch.tensor([3, 4]))


def test_wrapper_guards_raise_and_dump(tmp_path):
    pytest.importorskip("hepattn")
    from models.maskformer_model import HGCMaskformer

    wrapper = HGCMaskformer("t", torch.nn.Linear(2, 2), {"initial": 1e-4}, nonfinite_dump_dir=str(tmp_path))
    inputs, targets = make_batch()
    wrapper.on_validation_batch_start((inputs, targets), 3)
    losses = {"final": {"regression": {"l1": torch.tensor(NAN)}}}
    with pytest.raises(AssertionError) as err:
        wrapper.aggregate_losses(losses, stage="val")
    msg = str(err.value)
    assert "'final/regression/l1': nan" in msg and "batch event numbers: [10, 11, 12]" in msg
    dumps = list(tmp_path.glob("nonfinite_val_step0_batch3.pt"))
    assert dumps and str(dumps[0]) in msg
    saved = torch.load(dumps[0])
    assert list(saved["losses"]) == ["final"] and math.isnan(saved["losses"]["final"]["regression"]["l1"])
    assert torch.equal(saved["targets"]["event_number"], targets["event_number"])

    # a NaN among valid targets is caught when the batch arrives
    targets["particle_eta"][2, 0] = NAN
    with pytest.raises(AssertionError, match="'target particle_eta' non-finite for 1 events \\(event numbers: 12\\)"):
        wrapper.on_train_batch_start((inputs, targets), 0)
