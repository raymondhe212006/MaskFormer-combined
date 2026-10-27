# Toy-calorimeter MaskFormer trainings (reproduction of Qasim et al. 2022)

Branch `toy-calorimeter`: everything needed to train the HGCAL MaskFormer on rechits of the
`hg_toy_calorimeter` (https://github.com/shahrukhqasim/hg_toy_calorimeter; calibration data https://github.com/shahrukhqasim/hg_toy_calorimeter_data) (Qasim et al., *End-to-end multi-particle reconstruction in high occupancy imaging
calorimeters with graph neural networks*, Eur. Phys. J. C 82, 753 (2022), https://arxiv.org/abs/2204.01681),
with the paper's geometry, calibration, noise, event recipe and sample sizes.

## Data (group space on cmsdata)

`root://cmsdata.phys.cmu.edu//store/group/hgcal-ml-reconstruction/training_datasets/toy_calorimeter/<set>/`, NanoML parquet
(`RecHitHGC_*`, `SimCluster_*`), one `SHA256SUMS` per set, a `README.md` and `MANIFEST2.md` at the top level.

| set | events | content | per event |
|---|---|---|---|
| `pufree` | 5,000 (5 files) | 60 Type A single particles, no pileup | 21,052 ± 1,767 hits, 74.3 SimClusters (max 28,566 / 93) |
| `pu` | 5,000 (5 files) | 200 minbias in random 30° φ windows + 60 Type A | 34,453 ± 2,143 hits, 308 SimClusters (max 42,386 / 363) |
| `pufree_sec` | 15,000 sector rows | `pufree` cut into three 120° sectors (see below) | 7,017 ± 1,451 hits, 25.7 SimClusters (max 14,133 / 50) |
| `pufree_sec_rot60` | 15,000 sector rows | same with the sector boundaries rotated by 60° | max 12,443 / 49 |
| `test_typeB` | 19,999 | one Type B particle (e−/γ/π+, 1 mm in front of the face), no pileup | 2,532 ± 239 hits, 1 SimCluster |
| `test_typeB_sec` | 59,997 sector rows | `test_typeB` in sector form | |
| `photon_scan/E<E>` | 1,100 per point | single photons at 2, 3, 5, 10, 25, 50, 75, 100, 200, 300 GeV | `eval_sets/photons_E<E>` links for `jobs/eval_cmu_set.sh` |
| `photon_scan_sec` | | sector form of the scan | |
| `test_pu40` | 6,800 | PU40+1 (40 full minbias + probe) | 41,833 ± 7,897 hits, 948 SimClusters (max 73,940 / 1,569) |
| `test_pu200` | 6,800 | PU200+1 (200 full minbias + probe) | 166,323 ± 12,127 hits, 2,816 SimClusters (max 224,280 / 3,054) |

Truth: `SimCluster_boundaryEnergy` is the true energy, `SimCluster_truthAssistedEnergy` the sum of the particle's
own calibrated hits (the paper's truth-assisted baseline), `SimCluster_nHits` its hit count; noise hits have
`RecHitHGC_SimClusterBestMatchIdx == -1`. There is no `time` (the `-99` sentinel) and no trackster collection.

Sector sets (`--sectors 3 --sector-target deposited --sector-min-share 0.1`): one row per (event, sector),
boundaries at φ = 0/120/240°, hits assigned by sensor azimuth; a particle is a target in a sector only if ≥ 10 %
of its hit energy is there, its regression target is the energy deposited in that sector, and
`SimCluster_fullEnergy` keeps the full energy. `sector_has_leading` marks the sector holding most of the leading
particle's energy.
Sector sets have three rows per event: `pufree_sec` (15,000 rows) holds the same 5,000 events as `pufree`,
`test_typeB_sec` the same 19,999 events as `test_typeB`, and `photon_scan_sec` 3,300 rows per energy point.

## Code

- `src/scripts/dataset/toy_to_parquet.py`: `hg_toy_calorimeter` event sets (ra_pickles) → parquet; sector option.
- `src/scripts/dataset/toy_sectors.py`: the sector rule (numpy only).
- `src/scripts/dataset/compute_norm_par.py --identity VAR`: scale-dict fit with identity entries for absent inputs.
- `configs/toy_*_var_transform*.yaml`: scale dicts fitted on the production sets (plain, `_pu`, `_sec`).
- `configs/toy_rechit_pufree.yaml` (training A1: PU-free, full events, 512 queries, 32,768 padded nodes),
  `configs/toy_rechit_pufree_sec.yaml` (training A2: PU-free sector events, 128 queries, 16,384 nodes),
  `configs/toy_rechit_pu.yaml` (PU training, as planned; 1,536 queries × 65,536 nodes does not fit a quarter card:
  512 × 43,008 does). All three use the issue-5 configuration (`models.tasks.HGCIncidenceBasedRegressionTask`
  with `charged_classes: []`, `require_nodes_for_valid: true`, regression cost 1.0) and no `time` input.
  Every deviation from `configs/base_rechit_simcluster.yaml` is listed in the comment block at the top of each file.

## Running

Set `data.data_dir` to the local copy of the set (the configs carry the author's paths), then

```
python src/scripts/train/train_maskformer.py fit --config configs/toy_rechit_pufree.yaml \
    --trainer.logger.init_args.online false
```

GPU memory at batch 1 is ≈ 0.6 GiB + 0.64 GiB per 10⁶ (batch × queries × padded nodes): A1 ≈ 12 GB, A2 ≈ 2 GB.
Evaluate with `jobs/eval_cmu_set.sh` (mask-only matching, `EVAL_MATCH=mask`) on `eval_sets/photons_E<E>` and
`test_typeB`; the truth-assisted energy gives the detector-limit resolution curve.
