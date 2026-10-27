#!/bin/bash
#SBATCH --job-name=hgcal_eval
#SBATCH --output=logs/slurm/hgcal_eval_%x_%j.out
#SBATCH --partition=work
#SBATCH --qos=light
#SBATCH --time=06:00:00
#SBATCH --gres=mps:25
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

# Evaluates one mono-energetic single-particle set (a directory of parquet files) in a single process,
# one job per (particle, energy). A 1k-event trackster set needs ~1.2 GB RSS and ~3 min on an mps:25 share.
#
# Usage: sbatch -J <particle>_E<E> jobs/resolution/eval_cmu_set.sh <particle> <energy> [tag]
# Environment (all optional):
#   RUN_DIR    model directory holding config.yaml and ckpts/        (default: models/<RUN_NAME> under the checkout)
#   CKPT       checkpoint to use (default: ckpts/best-epoch=*.ckpt, else the lowest val_loss in ckpts/)
#   EVAL_SETS  directory with one sub-directory per set, <particle>_E<E>/, containing the parquet files
#              (default: $HOME/ml4reco/data/eval_sets; symlinks to the batch files are fine)
#   ATTN       encoder attention backend (default: torch; flash-varlen returns bf16 into the fp32 out_proj
#              under --trainer.precision 32-true, see issue #8)
# Outputs in $RUN_DIR/eval/<particle>_E<E>[_<tag>]/: <ckpt_stem>__test_<set>.parquet, *_losses.json, plots/, eval.log
set -u
PARTICLE=${1:?usage: sbatch jobs/resolution/eval_cmu_set.sh <particle> <energy> [tag]}
ENERGY=${2:?usage: sbatch jobs/resolution/eval_cmu_set.sh <particle> <energy> [tag]}
TAG=${3:-}
SET=${PARTICLE}_E${ENERGY}; NAME=${SET}${TAG:+_$TAG}

BASE_PATH=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
RUN_DIR=${RUN_DIR:-$BASE_PATH/models/${RUN_NAME:-hgcal_v2_trackster_simtrackster_20261012-T200958}}
EVAL_SETS=${EVAL_SETS:-$HOME/ml4reco/data/eval_sets}
TEST_DIR=$EVAL_SETS/$SET
APPTAINER_IMAGE=${APPTAINER_IMAGE:-/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest}
OUT_DIR=$RUN_DIR/eval/$NAME

cd "$BASE_PATH" || exit 1
if [ -z "${CKPT:-}" ]; then
  CKPT=$(ls "$RUN_DIR"/ckpts/best-epoch=*.ckpt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || CKPT=$(ls "$RUN_DIR"/ckpts/epoch=*-val_loss=*.ckpt 2>/dev/null | sed -E 's/(.*val_loss=)([0-9.]+)\.ckpt/\2 \1\2.ckpt/' | sort -n | head -1 | cut -d' ' -f2-)
fi
[ -f "$CKPT" ] || { echo "ERROR: no checkpoint found (RUN_DIR=$RUN_DIR, CKPT=$CKPT)"; exit 1; }
[ "$(ls "$TEST_DIR"/*.parquet 2>/dev/null | wc -l)" -ge 1 ] || { echo "ERROR: no parquet files in $TEST_DIR"; exit 1; }
mkdir -p "$OUT_DIR" logs/slurm

echo "job ${SLURM_JOB_ID:-local} on $(hostname): $NAME"; echo "ckpt:  $CKPT"; echo "data:  $TEST_DIR"; ls -lL "$TEST_DIR"
nvidia-smi >/dev/null 2>&1 && nvidia-smi | head -15
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# default_root_dir / data_dir / split overrides: the config may have been written in another tree (as in eval_lxplus.sh)
apptainer exec --nv \
  --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  --env LD_LIBRARY_PATH=/host-nvidia-libs:${LD_LIBRARY_PATH:-} \
  --env PYTHONPATH=${PYTHONPATH_EXTRA:+$PYTHONPATH_EXTRA:}$BASE_PATH/src \
  -B "$HOME" -B /cvmfs -B /tmp -B /usr/lib/x86_64-linux-gnu:/host-nvidia-libs \
  "$APPTAINER_IMAGE" \
  python src/scripts/evaluate/evaluate_maskformer.py test \
    -c "$RUN_DIR/config.yaml" \
    --ckpt_path "$CKPT" \
    --trainer.default_root_dir "$RUN_DIR" \
    --data.data_dir null --data.split null \
    --data.test_path "$TEST_DIR" \
    --data.test_suff "$NAME" \
    --data.num_workers 0 \
    --trainer.precision 32-true \
    --matmul_precision highest \
    --model.model.init_args.encoder.init_args.attn_type "${ATTN:-torch}" \
    --plot_dir "$OUT_DIR/plots" \
  2>&1 | tee "$OUT_DIR/eval.log"
RC=${PIPESTATUS[0]}
shopt -s nullglob
for f in "$RUN_DIR"/ckpts/*__test_"$NAME".parquet "$RUN_DIR"/ckpts/*__test_"$NAME"_losses.json; do mv "$f" "$OUT_DIR/"; done
shopt -u nullglob
[ "$RC" -ne 0 ] && { echo "[$NAME] FAILED with exit code $RC (log: $OUT_DIR/eval.log)"; exit "$RC"; }
echo "[$NAME] done -> $OUT_DIR"
