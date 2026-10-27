#!/bin/bash
#SBATCH --job-name=compute_norm_par
#SBATCH --output=logs/slurm/compute_norm_par_%j.out
#SBATCH --partition=work
#SBATCH --qos=light
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# Full-scale normalization-parameter scan (~460M rechits over 11 files).
# Writes TWO scale dicts: the node dict fitted on TOPO (input features) and the particle
# dict fitted on PART (regression targets); see docs/NORMALIZATION_FIX.md.
# This must NOT be run on the gateway: it kills processes above 60% CPU.
# Usage:  sbatch jobs/norm_par_job.sh [extra args for compute_norm_par.py]
#         TOPO=LayerCluster PART=SimCluster sbatch jobs/norm_par_job.sh   # LC -> SimCluster configs
#         TOPO=RecHitHGC   PART=SimCluster sbatch jobs/norm_par_job.sh   # rechit -> SimCluster configs
#         TOPO=ticlTrackstersCLUE3DHigh PART=ticlSimTracksters DATA_DIR=<dir with TICL tables> sbatch jobs/norm_par_job.sh
# Defaults (TOPO=RecHitHGC, PART=MergedSimCluster) serve configs/base.yaml.

set -euo pipefail

# Under sbatch the script runs from a copy in Slurm's spool, so BASH_SOURCE is useless
# here: prefer the submission directory, and allow an explicit override.
BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ ! -f "$BASE_DIR/src/scripts/dataset/compute_norm_par.py" ]]; then
  echo "ERROR: BASE_DIR=$BASE_DIR is not the hgcal-offline-reco root." >&2
  echo "Submit from that directory, or run: BASE_DIR=/path/to/hgcal-offline-reco sbatch jobs/norm_par_job.sh" >&2
  exit 2
fi
DATA_DIR="${DATA_DIR:-/home/export/mmarcheg/store/ml4reco/data_maskformers/parquet_out_single-files}"
TOPO="${TOPO:-RecHitHGC}"
PART="${PART:-MergedSimCluster}"
# output file names follow DEFAULT_OUTPUT in compute_norm_par.py
case "$TOPO" in RecHitHGC) NODE_OUT=hgcal_rechit_var_transform.yaml;; LayerCluster) NODE_OUT=hgcal_layercluster_var_transform.yaml;; ticlTrackstersCLUE3DHigh) NODE_OUT=hgcal_trackster_clue3dhigh_var_transform.yaml;; *) echo "unknown TOPO=$TOPO" >&2; exit 2;; esac
case "$PART" in MergedSimCluster) PART_OUT=hgcal_mergedsimcluster_var_transform.yaml;; SimCluster) PART_OUT=hgcal_simcluster_var_transform.yaml;; ticlSimTracksters) PART_OUT=hgcal_simtrackster_var_transform.yaml;; *) echo "unknown PART=$PART" >&2; exit 2;; esac
IMAGE="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest"

mkdir -p "$BASE_DIR/logs/slurm"
echo "Job running on $(hostname), data: $DATA_DIR, topoclusters: $TOPO -> configs/$NODE_OUT, particles: $PART -> configs/$PART_OUT"

apptainer exec -B /home/export/mmarcheg -B /cvmfs -B /tmp "$IMAGE" \
  python "$BASE_DIR/src/scripts/dataset/compute_norm_par.py" \
    --input-dir "$DATA_DIR" \
    --topocluster-collection "$TOPO" \
    --particle-collection "$PART" \
    --output "$BASE_DIR/configs/$NODE_OUT" \
    --particle-output "$BASE_DIR/configs/$PART_OUT" \
    --plot-dir "$BASE_DIR/plots/norm_par" \
    --threads 2 \
    "$@"
