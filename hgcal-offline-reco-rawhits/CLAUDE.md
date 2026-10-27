# CLAUDE.md: toy-calorimeter MaskFormer work (branch `toy-calorimeter`), status for an agent

Read `README.md` ("Toy calorimeter") and `docs/toy_calorimeter.md` first. This file is the state of the work as of
2026-10-23; it is a summary, not a plan. The authoritative plan and running state live outside the repository
(`~/ml4reco/toy_replication/PLAN.md` on the CMU cluster, with REPORTS*.md and MANIFEST*.md).

## Goal
Reproduce the data of Qasim et al. 2022 (arXiv:2204.01681, GravNet + object condensation on a toy HGCAL) and train the
hepattn-based MaskFormer of this repository on its rechits, so the two can be compared on identical samples. First
physics check: photon energy resolution (σ/E from a Gaussian fit of E_pred per fixed-energy point, 2 to 300 GeV)
against the paper's truth-assisted limit (the photon's own calibrated hits, noise excluded). Later: the paper's
EIOU ≥ 0.5 efficiency / fake / pT-response metrics on single particles with 0, 40 and 200 pileup, and an
offline → online distillation using the 120° trigger-sector models as teachers.

## Rules in force
- hepattn (the container's, 0.2.0) is never modified; anything we need is a subclass under `src/`
  (`models.tasks.HGCIncidenceBasedRegressionTask` with `charged_classes: []`, from PR #16 merged into master).
- Evaluation matches truth to queries on class + mask only (`EVAL_MATCH=mask`); the regression cost in the matching
  made every earlier resolution optimistic (issue #5).
- Hit-less SimClusters are not targets (`require_nodes_for_valid: true`); on the toy every SimCluster has ≥ 1 hit.
- Every training config lists its deviations from `configs/base_rechit_simcluster.yaml` in its header comment.

## Data production (done; phases 1 and 2)
Simulations with `hg_toy_calorimeter` (https://github.com/shahrukhqasim/hg_toy_calorimeter; calibration data https://github.com/shahrukhqasim/hg_toy_calorimeter_data) at the paper's geometry (commit with 778,712 sensors; the repository head has
3.1 M and is NOT the paper): Type A single particles e−/γ/π+/π0/τ, E ∈ [0.1, 200] GeV, η ∈ [1.4, 3.1] (310,000);
Type B singles started 1 mm in front of the face, e−/γ/π+, η ∈ [1.6, 2.9] (80,000); Pythia8 minimum bias (310,000 train
+ 200,000 test); 900 CPU tasks. Calibration: the paper's single global factor refitted on 240 photons (g = 2.992185;
the authors' recorded fit reproduced to 6 digits). Events: calibrated deposits + Gaussian noise + threshold, truth per
hit = particle with the largest deposit, close-by showers merged, particles kept with ≥ 1 hit. Hit counts match the
paper's Table 1 within errors; particle counts are 5 to 12 % low with pileup (the merge rule; documented).
Sets (parquet, on cmsdata): PU-free training 5,000 events (21,052 ± 1,767 hits, 74 SimClusters), PU training 5,000
(200 minbias in random 30° φ windows + 60 singles: 34,453 ± 2,143 hits, 308 SimClusters), Type B test 19,999,
PU40+1 6,800 (41,833 hits, 948 SimClusters, max 73,940 / 1,569), PU200+1 6,800 (166,323 hits, 2,816 SimClusters,
max 224,280 / 3,054), photon scan 10 × 1,100, and the sector forms (3 rows per event; deposited-energy target;
a particle is a target in a sector only with ≥ 10 % of its energy there, which relabels 0.7 % of the energy as noise).
Scale dicts fitted per set (`configs/toy_*_var_transform*.yaml`).

## Trainings (phase 3), status 2026-10-23
| run | data | queries × padded nodes | GPU memory (batch 1) | status |
|---|---|---|---|---|
| (A1) PU-free, full events | `pufree`, split 4 files train / 1 val | 512 × 32,768 | 11.7 GB | running, epoch ~41 of ≤ 100, val/loss 2.99 |
| (A3) sectored decoder, HEPTv2 (https://arxiv.org/abs/2606.20437, https://github.com/Graph-COM/HEPTv2)-style | `pufree` + per-hit sector labels | 3 × (128 × 16,384) | 5.8 GB | running, epoch ~44, val/loss 3.95 (code on branch `a3-sectored-decoder`, not on this branch) |
| (A2) sector samples | `pufree_sec` (+ `_rot60`) | 128 × 16,384 | ~2 GB | queued behind whichever finishes first |
| (B) PU, full events | `pu` | 512 × 43,008 planned | ~15 GB | after the three above |
Each run is followed by evaluations on `test_typeB` and the ten photon points with mask-only matching.
Open physics item from (A1)'s in-training checks (1,000 Type B events): the class head is fine (99 % valid, sane
confusion) but the shower is duplicated over several queries: at epoch 30, 21.6 fake queries per event, the matched
query holds 31 % of the shower energy, E_pred/E_true median 0.54. The (A3) smoke at the same stage kept 85 to 90 % on
the matched query with 2.7 fakes per event. Whether (A1) converges out of this is decided by its later checkpoints.

## GPU memory: measured law and what it rules out
On the quarter-card MPS slots (20,480 MiB) at batch 1: memory ≈ 616 MiB + 659 MiB per 10⁶ entries of
batch × queries × padded nodes (three measured points within 1 %). The decoder's query × node tensors dominate;
hit count itself does not (the encoder is linear). Consequences:
- (A1) at 50,000 padded nodes = 17.5 GB (85 % of the slot); at batch 2 it was killed (34 GB). 32,768 nodes = 11.7 GB.
- (B) with 512 queries × 43,008 nodes = 15 GB fits; the originally planned 1,536 × 65,536 (≈ 65 GB) does not fit any
  MPS share below a full card.
- Evaluation on PU40+1 at full coverage needs 1,570 queries × 73,941 nodes ≈ 76 GB; PU200+1 needs 3,055 × 224,281
  ≈ 450 GB. Neither is possible unsectored. Per 120° sector PU200+1 still has ~1,220 particles (> 1,024 queries);
  60° sub-sectors give ~720 particles and ~37,000 hits, which a 512-query model cannot cover either, so PU200+1
  evaluation needs either the (A3) per-sector decoder with ≥ 1,024 queries per sector or finer sectors.
  Also the number of learned queries is fixed at training time: a 512-query model cannot be evaluated with more.
- Padding is per dataset (`max_nodes`, `num_objects`); events above either limit are dropped. Dynamic per-batch
  padding would make memory follow the mean event and is a small dataset change not yet made.

## Known upstream bug (not on this branch)
hepattn's `mask_bce_loss` / `mask_dice_loss` broadcast the valid objects of all rows against every row's padding mask,
so with `batch_size > 1` and unequal padding each object is mis-normalised (−11.5 % BCE in a 2-row reproducer).
All HGCAL configs on master train at batch 4 to 32 and are affected; the toy runs at batch 1 are not. Fix as a subclass
(`HGCObjectHitMaskTask`, on `a3-sectored-decoder`); an issue draft exists, not yet posted.

## Backlog
Noise filter before the decoder (hepattn `HitFilterTask`; the toy's single-particle events are 87 % noise hits),
decided after (A1)'s photon scan; online-like (L1 trigger) emulation of the toy rechits for the distillation student;
PU40+1 / PU200+1 evaluation strategy per the memory limits above.
