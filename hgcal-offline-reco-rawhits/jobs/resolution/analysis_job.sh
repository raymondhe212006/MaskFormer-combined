#!/bin/bash
#SBATCH --job-name=hgcal_resolution
#SBATCH --output=logs/slurm/hgcal_resolution_%j.out
#SBATCH --partition=work
#SBATCH --qos=cpu_light
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Runs one of the analysis scripts of this directory inside the CVMFS image (they need awkward, scipy, matplotlib,
# mplhep). Usage: sbatch jobs/resolution/analysis_job.sh <script.py> [args...]
#   e.g. sbatch jobs/resolution/analysis_job.sh eval_gaussfit.py models/<run>/eval
#        sbatch jobs/resolution/analysis_job.sh plot_resolution.py models/<run>/eval photons     (RES_FIT / RES_FIT_EMAX / RES_HIDE_ABOVE env)
#        sbatch jobs/resolution/analysis_job.sh plot_sumts.py models/<run>/eval photons
set -u
SCRIPT=${1:?script name}; shift
HERE=$(realpath "$(dirname "${BASH_SOURCE[0]}")"); BASE_PATH=$(realpath "$HERE/../..")
APPTAINER_IMAGE=${APPTAINER_IMAGE:-/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest}
cd "$BASE_PATH" && mkdir -p logs/slurm
apptainer exec -B "$HOME" -B /cvmfs --env RES_FIT="${RES_FIT:-peak}" --env RES_FIT_EMAX="${RES_FIT_EMAX:-inf}" --env RES_HIDE_ABOVE="${RES_HIDE_ABOVE:-0}" \
  "$APPTAINER_IMAGE" python -I "$HERE/$SCRIPT" "$@"
