"""HGCAL-specific overrides of hepattn tasks.

hepattn is used unmodified; behaviour that HGCAL needs and hepattn does not offer is added here by inheritance.
"""

import torch
from hepattn.models.task import IncidenceBasedRegressionTask
from hepattn.utils.masks import topk_attn
from torch import Tensor


class HGCIncidenceBasedRegressionTask(IncidenceBasedRegressionTask):
    """IncidenceBasedRegressionTask with a configurable set of "charged" classes.

    hepattn's `get_proxy_feats` takes the proxy features of a query from the track branch whenever its predicted
    class index is < 3 (the PFlow convention: charged hadron / electron / muon). HGCAL inputs carry no tracks
    (`<input_constituent>_is_track == 0` everywhere), so every query predicted as a real class received zero proxy
    features and the regression loss pushed the classification head towards the null class (issue #5).

    Args:
        charged_classes: class indices whose proxy features come from the track branch. `None` keeps hepattn's
            default `[0, 1, 2]`; `[]` sends every query through the neutral (energy-weighted) branch, which is the
            setting for track-less inputs. All other arguments are passed to `IncidenceBasedRegressionTask`.
    """

    def __init__(self, *args, charged_classes: list[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.charged_classes = list(range(3)) if charged_classes is None else [int(c) for c in charged_classes]
        self.register_buffer(
            "_charged_class_idx", torch.tensor(self.charged_classes, dtype=torch.long), persistent=False
        )

    def is_charged_query(self, class_probs: Tensor) -> Tensor:
        pred_class = class_probs.argmax(-1)
        if len(self.charged_classes) == 0:
            return torch.zeros_like(pred_class, dtype=torch.bool)
        return torch.isin(pred_class, self._charged_class_idx.to(pred_class.device))

    def get_proxy_feats(
        self,
        incidence: Tensor,
        inputs: dict[str, Tensor],
        class_probs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Identical to hepattn.models.task.IncidenceBasedRegressionTask.get_proxy_feats except for the
        # charged/neutral split, which uses `self.charged_classes` instead of the hard-coded `< 3`.
        proxy_feats = torch.cat(
            [inputs[self.input_constituent + "_" + field].unsqueeze(-1) for field in self.fields],
            dim=-1,
        )

        charged_inc = incidence * inputs[self.input_constituent + "_is_track"].unsqueeze(1)
        charged_inc_top2 = (topk_attn(charged_inc, 2, dim=-2) & (charged_inc > 0)).float()
        charged_inc_max = charged_inc.max(-2, keepdim=True)[0]
        charged_inc_new = (charged_inc == charged_inc_max) & (charged_inc > 0)
        zero_track_mask = charged_inc_new.sum(-1, keepdim=True) == 0
        charged_inc = torch.where(zero_track_mask, charged_inc_top2, charged_inc_new)

        is_charged = self.is_charged_query(class_probs)

        proxy_feats_charged = torch.bmm(charged_inc, proxy_feats)
        proxy_feats_charged[..., 0] = proxy_feats_charged[..., 1] * torch.cosh(proxy_feats_charged[..., 2])
        proxy_feats_charged = self.scale_proxy_feats(proxy_feats_charged) * is_charged.unsqueeze(-1)

        inc_e_weighted = incidence * proxy_feats[..., 0].unsqueeze(1)
        inc_e_weighted *= 1 - inputs[self.input_constituent + "_is_track"].unsqueeze(1)
        inc = inc_e_weighted / (inc_e_weighted.sum(dim=-1, keepdim=True) + 1e-6)

        proxy_feats_neutral = torch.einsum("bnf,bpn->bpf", proxy_feats, inc)
        proxy_feats_neutral[..., 0] = inc_e_weighted.sum(-1)
        proxy_feats_neutral[..., 1] = proxy_feats_neutral[..., 0] / torch.cosh(proxy_feats_neutral[..., 2])

        proxy_feats_neutral = self.scale_proxy_feats(proxy_feats_neutral) * (~is_charged).unsqueeze(-1)
        proxy_feats = proxy_feats_charged + proxy_feats_neutral

        return proxy_feats, is_charged
