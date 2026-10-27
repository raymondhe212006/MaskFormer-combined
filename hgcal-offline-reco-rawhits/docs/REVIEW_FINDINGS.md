# Repo-wide review findings (branch `norm-fix`)

Bugs found in a full-repository review carried out after the normalization port, **beyond** the
normalization work itself. Normalization status, changes and their follow-ups live in
`docs/NORMALIZATION_FIX.md`; this file holds everything else. Every item below was verified against the
code and, where relevant, against the `100tau` sample; line numbers refer to `norm-fix` at `8ea5145` and to
the hepattn fork pinned in `pyproject.toml` (`mmarchegiani/hepattn@python_3_12_3`).

Ordered by impact.

## 1. The regression proxy is miswired for this repo's field order and for a detector without tracks

**Where:** hepattn `IncidenceBasedRegressionTask.get_proxy_feats` (pinned fork lines 1623, 1631–1632) and
`forward` (line 1510), used by every training config via `hepattn.models.task.IncidenceBasedRegressionTask`.

**What:** the proxy code hard-codes positions `0 = e, 1 = pt, 2 = eta` — the CLIC pflow convention,
`fields: [e, pt, eta, sinphi, cosphi]` — and does

```
proxy_charged[..., 0] = proxy_charged[..., 1] * cosh(proxy_charged[..., 2])
proxy_neutral[..., 1] = proxy_neutral[..., 0] / cosh(proxy_neutral[..., 2])
```

Every training config here declares `fields: ["e", "eta", "sinphi", "cosphi", "r"]`, so position 1 is **η**
and position 2 is **sinφ**.

- **Neutral candidates:** the η proxy is overwritten with `e_sum / cosh(sinphi)`, an energy-valued number
  that is then scaled with the particle `eta` transform (20 GeV → ≈ 6.9 while targets are in [−1, 1]).
- **Charged candidates** (`class_probs.argmax < 3`: charged hadrons, electrons, muons — the bulk of τ decay
  products): the proxy is built from `inputs["node_is_track"]`, which `HGCDataset` sets to **all zeros**
  (there are no tracks). Their proxy is therefore `scale(0)` per field: `e → −1.01`, `eta → ≈ 0`,
  `r → −12.4`, `sinphi/cosphi → 0`.
- With `mode: scale` (`preds = net(x) * proxy_feats`, line 1510) the η prediction of every charged particle
  is multiplied by ≈ 0 and **cannot be learned**; `r` is inflated 12×; sinφ/cosφ are zeroed.

Only `configs/test.yaml` uses the CLIC order (`["e", "pt", "eta", "sinphi", "cosphi"]`), and it would
`KeyError` on the missing `particle_pt` target.

**Options:** (a) `mode: offset` + `use_incidence: false` — proxies become zeros and are *added*, so the head
regresses directly; the safe fallback. (b) a repo-side task subclass whose proxy is built for
`[e, eta, sinphi, cosphi, r]` and has no track branch. (c) fix the indexing upstream with
`self.fields.index(...)`. **Do not** combine `use_incidence: false` with `mode: scale`: that multiplies by
zero and kills the head outright.

**Check:** train one epoch and compare `val/regression_eta_*` for charged vs neutral truth; under the current
wiring charged-particle η residuals will not improve.

## 2. `train_maskformer.py` loads another user's config as defaults

**Where:** `src/scripts/train/train_maskformer.py`, `config_dir = "/home/export/mmarcheg/ml4reco/hgcal-offline-reco/configs"`,
passed as `default_config_files` for `fit`.

**What:** jsonargparse applies `default_config_files` *under* the user's own `--config`. Where that path exists
(any user on the cluster), any key not set in your config silently takes the value from that user's
`base.yaml`, which may change without notice. Where it does not exist it is skipped, so the failure mode is
silent either way. Use `pathlib.Path(__file__).parents[3] / "configs"` (the commented-out line above it, with
the corrected depth) or drop the default.

## 3. `dummy_data: true` crashes

**Where:** `src/dataset/dataset_maskformers.py` `__getitem__` calls `self._generate_dummy_data(idx)`; no such
method exists on `master`/`norm-fix`.

**What:** any config with `data.dummy_data: true` raises `AttributeError` on the first item. A working
implementation exists on the `dev` branch (stashed with the dev normalization work); porting it would also let
`verify_norm.py` run without data.

## 4. Stale standalone callers

- `tests/test_dataset.py` constructs `HGCDataset` without the required `inputs` argument and reads from a
  hard-coded EOS path; it predates the collection refactor and cannot run.
- `src/scripts/dataset/read_HGCDataset.py` hard-codes another user's AFS config path
  (`/afs/cern.ch/work/m/mmarcheg/…/test_ticl_collections.yaml`).

## 4b. Incidence target is row-normalised, the prediction is column-normalised

**Where:** `src/dataset/dataset_maskformers.py` `load_event` (`inplace_csr_row_normalize_l1`, i.e. each *particle*
row sums to 1) versus hepattn `IncidenceRegressionTask.forward`, which applies `softmax(dim=1)` over the
*queries* for each node (each node column sums to 1), and `kl_div_loss = -true * log(pred)`.

**What:** with one-hot best-match targets the per-node normalisation is the identity, so the original
(HGPflow) convention gives a 0/1 target; the row normalisation instead turns every entry into
`1/N_nodes(particle)`, which is what `incidence_cutval` then thresholds (hence the "high fraction of masked
associations" warning on particles with many hits) and what the KL term compares against a per-node
distribution. The (`ticlTrackstersCLUE3DHigh`, `ticlSimTracksters`) config sets `normalize_incidence: false`
so its energy-share target keeps the per-node normalisation; the hit and layer-cluster configs still
row-normalise. Decide on one convention and align the cut value with it.

## 5. Fixed on this branch

- `tests/test_dataset.py`, `src/scripts/dataset/read_HGCDataset.py`: `particle_scale_dict_path` was passed to
  `HGCDataset` without being assigned (`NameError`). Introduced by the norm-fix port itself; fixed in `8ea5145`.
- `dataset_maskformers.py` `load_event`: the `verbose=True` print referenced `shape_original`, whose only
  assignment is commented out (`NameError` on every event when debugging). Pre-existing; fixed in `8ea5145`.
- `dataset_maskformers.py` `__init__`: the noise/pileup flags indexed the particle `eventId`/`bunchCrossing`
  with `safe_idx = 0` for unmatched topoclusters, which raises an awkward `IndexError` in any event that has
  topoclusters but no particle (e.g. the empty endcap of an endcap-split event, present in
  `taus_fineCalo_split.parquet`: the rechit configs could not load it). The lookup now pads every event to at
  least one particle. Pre-existing; fixed with the trackster/SimTrackster pair (2026-09-18).
- `configs/test.yaml` pointed `scale_dict_path` at a non-existent `configs/hgcal_var_transform.yaml`;
  repointed to the rechit dict in `526f8c2`.

## 6. Checked and found consistent (no action)

- `InputNet input_size` matches `inputs.topocluster` in every config: 6 for the rechit configs (with `time`),
  5 for `base.yaml` and the LayerCluster configs.
- All 13 distinct `SimCluster_pdgId` values in `100tau_train` are in `HGCDataset.class_labels`, and none maps to
  the neutrino/unknown classes (−1/−2). That matters because `__getitem__` marks every `class < 5` particle
  valid, so a −1 label would reach the classification loss as an out-of-range target.
- LayerCluster `SimCluster_MatchIdx` never exceeds the event's particle count (max 76 vs 77 particles), so
  `compute_incidence_matrix` cannot index out of range on this sample.
- ~~`LayerCluster_SimCluster_MatchQual` ∈ [0, 1] with higher = better, so `reduce_to_best_match`'s `argmax` picks
  the intended candidate.~~ **Wrong; fixed 2026-09-21.** `MatchQual` is the layer-cluster → SimCluster association
  *score*, lower = better: on `taus_MergedCaloTruth/taus.parquet` (150 events, 10 219 multi-match layer clusters)
  the candidate with the *smallest* score is the SimCluster owning most of the cluster's rechit energy
  (`RecHitHGC_SimClusterBestMatchIdx`) for 94% of them, the largest for 2%; single-match clusters carry
  `MatchQual = 0` and the per-cluster sum of the candidates exceeds 1, so it is not a shared-energy fraction.
  `reduce_to_best_match` now takes the direction from `MANY_TO_MANY_MATCH_MAP[...]["best"]` (`"min"`) in
  `src/dataset/collections.py`, which the evaluation plots share. LayerCluster/SimCluster models trained before
  the fix used the wrong best match for the multi-match clusters (~19% of the layer clusters on that sample).
- Events with all-noise nodes: `fraction_masked_nonzero` divides by zero → NaN → the warning comparison is
  simply false; no crash.
- No non-positive rechit or layer-cluster energies in `100tau_train` (mins 0.004 / 0.007 GeV), so the new
  `fn: log` on node energy is safe on this sample (guard proposed in `NORMALIZATION_FIX.md` §6.6).
