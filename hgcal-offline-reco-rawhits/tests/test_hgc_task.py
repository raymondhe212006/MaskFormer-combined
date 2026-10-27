"""models.tasks.HGCIncidenceBasedRegressionTask: charged/neutral split configurable, hepattn untouched."""

import pathlib

import pytest
import torch
from torch import nn

hepattn_task = pytest.importorskip("hepattn.models.task")
from models.tasks import HGCIncidenceBasedRegressionTask  # noqa: E402

FIELDS = ["e", "eta", "sinphi", "cosphi", "r"]
SCALE_DICT = pathlib.Path(__file__).resolve().parents[1] / "configs" / "hgcal_simtrackster_var_transform.yaml"


def make(cls, **kw):
    torch.manual_seed(0)
    return cls(
        name="regression", input_constituent="node", input_object="query", output_object="pflow",
        target_object="particle", fields=FIELDS, loss_weight=10.0, cost_weight=1.0, scale_dict_path=str(SCALE_DICT),
        net=nn.Linear(4, len(FIELDS)), loss="l1", use_incidence=True, use_nodes=True, cost="new", **kw,
    )


@pytest.fixture(params=[0, 1], ids=["no_tracks", "tracks"])
def batch(request):
    B, P, N, C = 2, 7, 40, 4
    g = torch.Generator().manual_seed(1)
    inputs = {"node_" + f: torch.rand(B, N, generator=g) + 0.1 for f in FIELDS}
    inputs["node_is_track"] = (torch.rand(B, N, generator=g) < 0.3).float() * request.param
    inc = torch.rand(B, P, N, generator=g)
    probs = torch.rand(B, P, C, generator=g)
    return inputs, inc / inc.sum(-1, keepdim=True), probs


def proxy(task, batch):
    inputs, inc, probs = batch
    with torch.no_grad():
        return task.get_proxy_feats(inc, inputs, probs)


@pytest.mark.parametrize("charged_classes", [None, [0, 1, 2]])
def test_default_split_matches_hepattn(batch, charged_classes):
    ref = proxy(make(hepattn_task.IncidenceBasedRegressionTask), batch)
    ours = proxy(make(HGCIncidenceBasedRegressionTask, charged_classes=charged_classes), batch)
    assert torch.equal(ours[0], ref[0]) and torch.equal(ours[1], ref[1])


def test_empty_charged_classes_is_all_neutral(batch):
    inputs, inc, probs = batch
    feats, is_charged = proxy(make(HGCIncidenceBasedRegressionTask, charged_classes=[]), batch)
    assert not is_charged.any()
    # every query gets the neutral proxy, whatever its predicted class
    neutral_probs = torch.zeros_like(probs); neutral_probs[..., -1] = 1.0  # argmax = null -> neutral in hepattn
    ref, _ = proxy(make(hepattn_task.IncidenceBasedRegressionTask), (inputs, inc, neutral_probs))
    assert torch.equal(feats, ref)
    assert torch.isfinite(feats).all()


def test_subset_of_classes(batch):
    inputs, inc, probs = batch
    feats, is_charged = proxy(make(HGCIncidenceBasedRegressionTask, charged_classes=[2]), batch)
    assert torch.equal(is_charged, probs.argmax(-1) == 2)
    base_feats, base_charged = proxy(make(hepattn_task.IncidenceBasedRegressionTask), batch)
    # where the base class also calls the query charged, the proxies agree; neutral queries agree with the
    # all-neutral evaluation
    assert torch.equal(feats[is_charged], base_feats[is_charged])
    neutral_feats, _ = proxy(make(HGCIncidenceBasedRegressionTask, charged_classes=[]), batch)
    assert torch.equal(feats[~is_charged], neutral_feats[~is_charged])


def test_state_dict_unchanged():
    base = make(hepattn_task.IncidenceBasedRegressionTask)
    ours = make(HGCIncidenceBasedRegressionTask, charged_classes=[])
    assert set(base.state_dict()) == set(ours.state_dict())
