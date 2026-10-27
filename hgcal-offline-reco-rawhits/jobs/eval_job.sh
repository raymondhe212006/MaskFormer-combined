#!/bin/bash
#SBATCH --job-name=hgcal_maskformer_eval
#SBATCH --output=logs/slurm/hgcal_maskformer_eval_%j.out
#SBATCH --partition=work
#SBATCH --qos=light                # max mps:25 per job
#SBATCH --time=1-00:00:00
#SBATCH --gres=mps:25              # 25% of the GPU, the light qos per-job limit
#SBATCH --cpus-per-task=32         # Match CPU threads to GPU work
#SBATCH --mem=128G                 # Max 4G per CPU (32 CPUs x 4G = 128G limit)

# Evaluates a trained MaskFormer run on GPU with evaluate_maskformer.py, one test file at a time.
# Usage: sbatch jobs/eval_job.sh <run_dir> <test.parquet | file_list.txt> [extra evaluate_maskformer.py args...]
#   e.g. sbatch jobs/eval_job.sh logs/hgcal_v2_trackster_simtrackster_20261012-T200958 jobs/photon_eval_files.txt
#
# A .txt argument is a list of parquet files, one per line (blank lines and # comments ignored).
# Each file is evaluated in its own process, and its outputs go to <run_dir>/eval/<file_stem>/:
#   <ckpt_stem>__test_<file_stem>.parquet               predictions
#   <ckpt_stem>__test_<file_stem>_losses.json           loss summary
#   plots/                                              evaluation plots
# A failing file is reported and skipped; the job exits non-zero if any file failed.

BASE_PATH=/home/export/$USER/ml4reco/hgcal-offline-reco
RUN_DIR=${1:?usage: sbatch jobs/eval_job.sh <run_dir> <test.parquet | file_list.txt> [extra args...]}
INPUT=${2:?usage: sbatch jobs/eval_job.sh <run_dir> <test.parquet | file_list.txt> [extra args...]}
shift 2
EXTRA_ARGS=("$@")

cd $BASE_PATH
RUN_DIR=$(realpath "$RUN_DIR")

if [[ "$INPUT" == *.txt ]]; then
  mapfile -t TEST_FILES < <(grep -v -E '^\s*(#|$)' "$INPUT")
else
  TEST_FILES=("$INPUT")
fi

nvidia-smi
echo "CUDA_VISIBLE_DEVICES on host: ${CUDA_VISIBLE_DEVICES:-<not set>}"

APPTAINER_IMAGE=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest
echo "Evaluating $RUN_DIR on ${#TEST_FILES[@]} file(s) on GPU within Apptainer container..."
echo "Using Apptainer image: $APPTAINER_IMAGE"

# With --gres=mps:X allocations SLURM may not set CUDA_VISIBLE_DEVICES; fall back to GPU 0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FAILED=()
for TEST_PATH in "${TEST_FILES[@]}"; do
  NAME=$(basename "$TEST_PATH" .parquet)
  OUT_DIR=$RUN_DIR/eval/$NAME
  mkdir -p "$OUT_DIR"
  echo "=================================================================================="
  echo "[$NAME] $TEST_PATH -> $OUT_DIR"

  # The prediction writer puts <ckpt_stem>__test_<test_suff>.parquet next to the checkpoint;
  # it is moved into OUT_DIR once the run (and the plot chain reading it) has finished.
  apptainer exec \
    --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env LD_LIBRARY_PATH=/host-nvidia-libs:$LD_LIBRARY_PATH \
    --env PYTHONPATH=$BASE_PATH/src \
    -B /home/export/$USER \
    -B /cvmfs \
    -B /tmp \
    -B /usr/lib/x86_64-linux-gnu:/host-nvidia-libs \
    $APPTAINER_IMAGE \
    python src/scripts/evaluate/evaluate_maskformer.py test \
      -c "$RUN_DIR/config.yaml" \
      --data.test_path "$TEST_PATH" \
      --data.test_suff "$NAME" \
      --data.num_workers 0 \
      --trainer.precision 32-true \
      --matmul_precision highest \
      --plot_dir "$OUT_DIR/plots" \
      "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/eval.log"
  RC=${PIPESTATUS[0]}

  shopt -s nullglob
  for f in "$RUN_DIR"/ckpts/*__test_"$NAME".parquet "$RUN_DIR"/ckpts/*__test_"$NAME"_losses.json; do
    mv "$f" "$OUT_DIR/"
  done
  shopt -u nullglob

  if [ "$RC" -ne 0 ]; then
    echo "[$NAME] FAILED with exit code $RC (log: $OUT_DIR/eval.log)"
    FAILED+=("$NAME")
  else
    echo "[$NAME] done -> $OUT_DIR"
  fi
done

echo "=================================================================================="
echo "Evaluated $(( ${#TEST_FILES[@]} - ${#FAILED[@]} ))/${#TEST_FILES[@]} file(s); outputs in $RUN_DIR/eval/"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed: ${FAILED[*]}"
  exit 1
fi
