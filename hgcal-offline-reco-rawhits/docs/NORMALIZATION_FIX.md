# Feature normalization: status, fixes on this branch, and how to validate them

Branch: `norm-fix` (off `master`). Written so that a reviewer — human or agent — can reproduce every
claim below with the commands given. All numbers come from the `100tau` sample
(`100tau_train.parquet`: 8000 events, 344,426,502 rechits, 3,178,040 layer clusters, 108,487 SimClusters;
checks on `100tau_test.parquet`: 1000 events) unless stated otherwise.

## 1. How normalization works here

Inputs and targets are scaled by `hepattn.utils.scaling.FeatureScaler`, one `VarTransform` per key, from a
YAML "scale dict":

```
x -> fn(x)                     fn in {none, sqrt, log, log1p}
x -> (x - shift) / scale       type=std:         shift=mean,        scale=std
                               type=min_max_sym: shift=(max+min)/2, scale=(max-min)/2
```

`fn` is applied **before** the shift/scale, so `mean/std/min/max` in the YAML must be statistics of the
transformed variable. Three consumers read scale dicts:

| consumer | what it scales | keys used |
|---|---|---|
| `HGCDataset.load_event` node features | topocluster inputs (`e`, `eta`, `r`, `t`) | `e eta r t` |
| `HGCDataset.load_event` particle_data | regression targets | `e eta r` (+ identity for `sinphi cosphi`) |
| `hepattn IncidenceBasedRegressionTask` | incidence-aggregated **proxy** particle features; output is `net(x) * proxy` (`mode: scale`) | `e eta cosphi sinphi r` |
| `HGCPredictionWriter` | inverse-transforms predicted particle fields | same as targets |

The last three must agree with each other (they define the regression space); the first is a different
population altogether.

## 2. Status on `master` before this branch (what was wrong)

### 2.1 Only a rechit dict existed, and every config used it — including the LayerCluster ones
`src/scripts/dataset/compute_norm_par.py` derived `RecHitHGC_*` branches only (`derive_rechit_arrays`), wrote
one file (`configs/hgcal_rechit_var_transform.yaml`), and all six configs pointed at it — including
`base_layercluster_simcluster.yaml` and `test_ticl_collections.yaml`. LayerCluster→SimCluster training therefore
ran on rechit-fitted transforms. Layer-cluster energies have a 21× larger spread than rechit energies (raw std
1.59 vs 0.076 GeV), so with the rechit `e` entry (`log1p`, mean 0.012, std 0.024) a 49 GeV layer cluster
scaled to `(log1p(49)-0.012)/0.024 ≈ 162`.

### 2.2 Regression targets were scaled with the node dict (shared key names)
`dataset_maskformers.py` did `particle_data = self.scaler.transform(particle_data)` with the same scaler used for
node features. Both populations use the keys `e`/`eta`/`r`. Measured raw spreads on `100tau_train`:

| key | RecHitHGC std | LayerCluster std | SimCluster std | ratio (rechit) | ratio (LC) |
|---|---|---|---|---|---|
| `e` | 0.076 GeV | 1.59 GeV | 24.7 GeV | **339×** | **15.6×** |
| `eta` | 2.48 | 2.21 | 2.26 | 1.2× | 1.0× |
| `r` | 21.4 cm | 11.5 cm | 9.9 cm | 2.1× | 1.2× |

With master's rechit dict, a SimCluster of median recEnergy 14.9 GeV became a target of
`(log1p(14.9)-0.012)/0.024 ≈ 115`, and a 106 GeV one ≈ 194, while inputs were O(1). Master's own script
already detected this (`report_particles` printed a WARNING about the shared keys) but did not fix it.
The prediction writer inverse-transformed predictions with the same node dict (`prediction_writer.py:48`).

### 2.3 The node `r` input feature was never scaled
`node_features["r"] = topo_R` entered the network raw (320–570 cm) while `e`/`eta` were scaled
(`dataset_maskformers.py`, `"r": torch.cat([topo_R])`).

### 2.4 `log1p` is nearly the identity for rechit and layer-cluster energies
Master had already fixed the older fn-before-stats bug (stats were once measured on raw `e` and tagged
`fn: sqrt`, collapsing every rechit energy to ≈ −0.998) and chose `fn: log1p`. But `log1p(x) ≈ x` for `x ≪ 1`,
and the rechit median is 0.006 GeV, the layer-cluster median 0.05 GeV. The script's own alternatives table
on `100tau_train` (rechits, 344M hits):

| fn | mean | std | max | nσ of max |
|---|---|---|---|---|
| none | 0.0144 | 0.0761 | 14.43 | 189 |
| sqrt | 0.1006 | 0.0654 | 3.80 | 57 |
| **log** | **−4.77** | **0.74** | 2.67 | **10** |
| log1p | 0.0132 | 0.0396 | 2.74 | 69 |

Skewness measured separately on `100tau_test`: raw 41, sqrt 11, log1p 24, **log 1.7** (rechits); for layer
clusters log1p 3.3 vs log 1.6; for SimClusters (median 1.6 GeV) log1p 0.64 — there `log1p` is fine, and `log`
is unsafe because `recEnergy` is exactly 0 for some SimClusters.

### 2.5 Latent crash: `FeatureScaler(..., strict=...)`
`HGCDataset` calls `FeatureScaler(scale_dict_path, strict=strict_scaling)`. No hepattn that this repo can
resolve has that parameter: not upstream `samvanstroud/hepattn` (any branch), not the fork pinned in
`pyproject.toml` (`mmarchegiani/hepattn@python_3_12_3`, checked 2026-09-09). Against those, constructing the
dataset raises `TypeError`. Either the cluster image carries a further-patched hepattn, or master's dataset
cannot be instantiated at all; `pip show hepattn` inside the image will tell.

## 3. What this branch changes

| file | change |
|---|---|
| `src/scripts/dataset/compute_norm_par.py` | Fits **two** dicts per run: `--topocluster-collection {RecHitHGC,LayerCluster}` → `--output`, `--particle-collection {MergedSimCluster,SimCluster}` → `--particle-output` (defaults from `DEFAULT_OUTPUT`). `derive_topocluster_arrays` mirrors `HGCDataset` per collection (LC takes eta/phi from the file; no time). `TOPO_VAR_SPECS` energy `fn: log`; new `PARTICLE_VAR_SPECS` (`e` log1p/std, `eta`/`r` min_max_sym with quantile bounds, identity `sinphi`/`cosphi`). `report_particles` → `report_shared_key_spreads`. `--input-dir` accepts a single file. `NSIGMA_FAIL["t"]` 25 → 75 (measured 53σ physical tail on 100tau; `t` uses quantile bounds so the tail never enters shift/scale). Refuses `--output == --particle-output`. `ru_maxrss` units fixed on macOS. |
| `src/dataset/dataset_maskformers.py` | `HGCDataset`/`HGCDataModule` take `particle_scale_dict_path`; `self.particle_scaler` scales `particle_data`; node `r` input scaled with `self.scaler.transforms["r"]`; `make_feature_scaler()` passes `strict` when hepattn supports it and falls back (with a warning) otherwise. |
| `src/callbacks/prediction_writer.py` | Un-scales predictions with `ds.particle_scaler`. |
| `configs/*.yaml` (6) | `particle_scale_dict_path: &particle_scale_dict_path …`; regression task `scale_dict_path: *particle_scale_dict_path`; LayerCluster configs point at `hgcal_layercluster_var_transform.yaml`. `test.yaml`'s node path pointed at a non-existent `hgcal_var_transform.yaml` and now points at the rechit file. |
| `configs/hgcal_layercluster_var_transform.yaml`, `configs/hgcal_simcluster_var_transform.yaml` | **new**, fitted on `100tau_train.parquet` (see §4). |
| `configs/hgcal_rechit_var_transform.yaml` | refit with `fn: log` on `100tau_train.parquet` (see §4). |
| `jobs/norm_par_job.sh` | `TOPO=… PART=… sbatch jobs/norm_par_job.sh` writes both dicts. |
| `src/scripts/dataset/verify_norm.py` | **new**: end-to-end check through `HGCDataset` (§5). |
| `tests/test_dataset.py`, `src/scripts/dataset/read_HGCDataset.py` | pass `particle_scale_dict_path`. |
| `README.md`, `.gitignore` (`plots/`, `logs/`) | documentation; ignore job outputs. |

Design choice kept from master: `type: std` for energy, `min_max_sym` with detector-envelope extremes for
geometry. Particle `eta`/`r` use **quantile** bounds because SimCluster impact points reach |η| ≈ 5.9 (outside
the HGCAL acceptance) while the bulk sits within ±3.1; with extremes the bulk would occupy the middle fifth
of [−1, 1].

## 4. Scale dicts shipped on this branch — provenance matters

| file | fitted on | sample | status |
|---|---|---|---|
| `hgcal_rechit_var_transform.yaml` | RecHitHGC | `100tau_train.parquet`, 344M hits (local) | refit here, `e: log` |
| `hgcal_layercluster_var_transform.yaml` | LayerCluster | `100tau_train.parquet`, 3.2M LCs (local) | new |
| `hgcal_simcluster_var_transform.yaml` | SimCluster | `100tau_train.parquet`, 108k SCs (local) | new |
| `hgcal_mergedsimcluster_var_transform.yaml` | MergedSimCluster | — | **missing**: the local sample has no `MergedSimCluster` collection. `configs/base.yaml`, `base_rechit_mergedsimcluster.yaml` and `test.yaml` reference it and will fail with `FileNotFoundError` until it is produced on the cluster: `sbatch jobs/norm_par_job.sh` (defaults are RecHitHGC/MergedSimCluster). |
| `hgcal_trackster_clue3dhigh_var_transform.yaml` | ticlTrackstersCLUE3DHigh | `taus_fineCalo_split.parquet` (TICLv5, CMSSW_20_0_0_pre1), 62,533 tracksters, all 20,000 endcap-split events | new (2026-09-18), for `configs/base_trackster_simtrackster.yaml` |
| `hgcal_simtrackster_var_transform.yaml` | ticlSimTracksters | same file, 579,188 SimTracksters | new (2026-09-18, `e` refit on `regressed_energy` 2026-09-21); fitted with `--allow-outliers` because 3 SimTracksters have their barycenter in the opposite endcap (`eta` −2.47, 10.9σ); quantile bounds keep them out of the fit |

Master's previous rechit dict was fitted on the cluster's `tau20to200` sample (11 files, 462M hits). The
geometry entries are detector envelopes and agree to <0.1 % between the two samples (e.g. `r` 323.7–544/569,
`eta` ±3.137); the `e` entry cannot be compared because the transform changed. **Before production training on
the cluster, regenerate all dicts from the cluster sample** with the three `sbatch` lines in the README so
that inputs, targets and the sample match. Every YAML header records its input path, so provenance is
auditable.

## 5. Validation procedure

Run from the repo root in an environment with hepattn, awkward, pyarrow (e.g. the cluster image, or
locally `…/miniconda3/envs/hepattn/bin/python`).

### 5.1 Static consistency (no data needed)
```bash
python - <<'EOF'
import yaml, glob
for f in sorted(glob.glob("configs/*.yaml")):
    c = yaml.safe_load(open(f)); d = c.get("data", {})
    if "scale_dict_path" not in d: continue
    def walk(o):
        if isinstance(o, dict):
            if "scale_dict_path" in o and "fields" in o: yield o["scale_dict_path"]
            for v in o.values(): yield from walk(v)
        elif isinstance(o, list):
            for v in o: yield from walk(v)
    task = list(walk(c.get("model", {})))
    print(f"{f:45s} node={d['scale_dict_path']:45s} particle={d['particle_scale_dict_path']:50s} task_ok={task==[d['particle_scale_dict_path']]}")
EOF
```
Expected: every line `task_ok=True`; node file matches `topocluster_collection`, particle file matches
`particle_collection`.

### 5.2 Fit smoke test (cheap, prints the alternatives table and the shared-key ratios)
```bash
python src/scripts/dataset/compute_norm_par.py --input-dir <file_or_dir> --topocluster-collection LayerCluster --particle-collection SimCluster --max-events 200 --no-plots --dry-run
python src/scripts/dataset/compute_norm_par.py --input-dir <file_or_dir> --topocluster-collection RecHitHGC   --particle-collection SimCluster --max-events 20  --no-plots --dry-run
```
Expected: two YAML blocks per run (node, particle); `Shared keys` section shows `e` ratio ≫ 10×;
`Alternative transforms for 'e'` shows `log` with the smallest `nsig_hi`. A 20-event rechit run may FAIL the
`t` gate on small samples — that is the outlier check working; the full sample passes at 53σ < 75.

### 5.3 End-to-end through the dataset (the important one)
```bash
python src/scripts/dataset/verify_norm.py --config configs/base_layercluster_simcluster.yaml --parquet <test.parquet> --num-events 300 --quiet
python src/scripts/dataset/verify_norm.py --config configs/base_rechit_simcluster.yaml       --parquet <test.parquet> --num-events 30  --quiet
```
Expected `ALL CHECKS PASSED: True`, exit 0. Reference output on `100tau_test`, LayerCluster→SimCluster
(300 events, 119,290 nodes, 4,078 particles):

```
node e     min=-1.631 max=3.871 mean=0.004 std=0.995   (was std 0.02 with the collapsed sqrt dict; 162 max with the rechit log1p dict)
node r     min=-0.999 max=0.035                        (was 320-570 raw)
part e     min=-1.009 max=2.029 mean=-0.004 std=1.001  (was O(100) with the node dict)
part eta   q1=-0.920 q99=0.926
part r     q1=-0.966 q99=0.705
```
The script fails if: node `e` std ≤ 0.1, node |`r`| > 1.5, particle `e` std outside (0.3, 3), particle
`eta`/`r` q1/q99 outside ±1.5, anything non-finite, the two dicts are the same file, or the regression task is
not given the particle dict.

### 5.4 Before/after histograms
`plots/norm_par/` holds the fit script's raw and scaled histograms for every variable of the three dicts shipped
here (`RecHitHGC_*`, `LayerCluster_*`, `SimCluster_*`; `<var>_before.png` raw, `<var>_after.png` after
`fn` + shift/scale, with the ±1 band marked). The `*_e_after.png` panels are the direct check that energy
is no longer a spike; `RecHitHGC_t_after.png` shows the fitted (valid-time) band only — the −99 sentinel is not
in the fit sample (§6.1). Regenerated by every `compute_norm_par.py` run unless `--no-plots`.

### 5.5 Training smoke test
```bash
python src/scripts/train/train_maskformer.py fit --config configs/base_layercluster_simcluster.yaml --data.num_workers 0 --trainer.num_sanity_val_steps 0 --trainer.max_epochs 1 --trainer.limit_train_batches 5 --trainer.limit_val_batches 2
```
Expected: the regression task constructs with the particle dict; `regression` losses are O(1) at step 0
rather than O(100). Then `evaluate_maskformer.py test` and check that `pflow_e` in the output parquet is in
GeV (the writer now inverse-transforms with the particle dict).

## 6. Normalization follow-ups (not done on this branch)

Ordered by impact. Bugs found elsewhere in the repo during the review that accompanied this work — including
the regression-proxy miswiring that sits directly on the target path — are in `docs/REVIEW_FINDINGS.md`.

1. **Time sentinel** (`configs/hgcal_rechit_var_transform.yaml` `t`): `t` is fitted on valid times
   (`t > −1`, 1.18 % of hits) but `load_event` transforms **all** values, so the −99 sentinel lands at
   `(−99 − 12.09)/1.35 ≈ −82` in scaled units for ~99 % of nodes — and the refit made this worse (the
   previous dict had scale ≈ 8 → −14.7). Both `base_rechit_*.yaml` configs feed `time`. Fix in `load_event`:
   transform only valid times and park the sentinel at a fixed bounded value (e.g. −2), or add a validity
   flag input (changes `InputNet input_size`). The norm script prints this WARNING on every run.
2. **Detector-envelope bounds now depend on the fitted sample.** The rechit geometry entries use
   `bounds="extremes"` *because* the envelope is fixed and reproducible, but the refit on one local file shrank
   them (x ±256 → [−168, 182]; `r` max 568.9 → 544.1). A rechit at `r = 568.9` cm (present in the cluster
   sample) now scales to 1.22 for the `r` input. Either hard-code the envelope in `TOPO_VAR_SPECS`
   (`x/y` ±256.33, `z` ±513.28, `rho` 31.84–256.71, `r` 323.73–568.90, `eta` ±3.137 — the values of the
   462M-hit cluster fit) and only *report* the measured extremes, or always fit on the full cluster sample (§4).
3. **Regenerate all dicts on the cluster sample** (§4) — one `sbatch` per (topocluster, particle) pair — and
   produce the missing `hgcal_mergedsimcluster_var_transform.yaml`. Until then `configs/base.yaml`
   (the default in `jobs/gpu_job.sh` and the README), `base_rechit_mergedsimcluster.yaml` and `test.yaml`
   fail at startup with `FileNotFoundError`. Existing checkpoints are invalid anyway since the target space
   changed.
4. **`strict` in `FeatureScaler`** (§2.5): no resolvable hepattn accepts it, so `make_feature_scaler`'s
   fallback means `strict_scaling=True` is currently a no-op — a key missing from either dict silently becomes
   an identity transform (exactly the failure class this branch fixes). Replace the fallback with an explicit
   assertion that the required keys (`e eta r [t]` on the node dict; the configured target fields on the
   particle dict) exist in `scaler.scale_dict`, and narrow the `except TypeError` (it also swallows a
   `TypeError` from `Path.open(None)`).
5. **Single source of truth for the regression space.** Four places must agree (`data.scale_dict_path`,
   `data.particle_scale_dict_path`, the task's `*particle_scale_dict_path` anchor in 7 configs, the writer's
   `ds.particle_scaler`). Add a startup assertion `task.scaler.scale_dict == dataset.particle_scaler.scale_dict`
   (or have the writer read the task's scaler), so a mis-anchored config fails loudly instead of training in
   mixed units.
6. **`fn: log` on node energy has no guard in the dataset.** `load_event` applies `torch.log(topo_e)`
   directly; the only domain check is in the fit script. No non-positive rechit or layer-cluster energies exist
   in `100tau_train` (mins 0.004 / 0.007 GeV; 225 SimClusters have `recEnergy == 0`, handled by `log1p`), but a
   single zero in another file would give `−inf` → NaN loss with no message. A `clamp_min` or an assertion on
   `full_data_array["topo_energy"]` in `__init__` makes the switch robust.
7. **Duplicated branch maps / kinematics** between `compute_norm_par.py` (`TOPO_FIELD_MAP`,
   `derive_topocluster_arrays`, `derive_particle_arrays`) and `dataset_maskformers.py` (`TOPO_FIELD_MAP`,
   `PARTICLE_SUFFIX_TEMPLATES`, the "Derive kinematics" block); they have already drifted trivially
   (`nhits`). Import from a small deps-free module instead.
8. **`mode: scale` × centred energy**: `type: std` energy crosses zero at the median, where a multiplicative
   correction cannot act; `mode: offset` is the natural pairing with centred scaling. See also
   `docs/REVIEW_FINDINGS.md` item 1, which makes the current proxy wrong regardless of the scaling.
9. **SimCluster targets outside acceptance** (|η| up to 5.9, `r` down to 224 cm): quantile bounds keep the
   bulk well scaled, but these particles may deserve a selection at dataset level.

## 7. History

The same fix was first implemented on the `dev` branch (which had diverged from master before the
`compute_norm_par.py` rewrite) and validated there; it is preserved as a stash on the local `dev` checkout.
This branch is a port to master's code structure (`TOPO_FIELD_MAP`/collection arguments, `VarSpec`-based fit).
