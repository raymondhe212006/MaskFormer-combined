#!/bin/bash
# Evaluates a trained MaskFormer run on an lxplus-gpu machine (no Slurm, runs on the local GPU),
# one test file at a time, inside the CVMFS apptainer image.
#
# Usage (from anywhere, on lxplus-gpu; use tmux/screen, the loop takes a while):
#   bash jobs/eval_lxplus.sh [model_dir] [file_list.txt] [extra evaluate_maskformer.py args...]
#
# Defaults:
#   model_dir     /eos/user/m/mmarcheg/ml4reco/maskformers/models/hgcal_v2_trackster_simtrackster_20261012-T200958
#   file_list     jobs/photon_eval_files.txt
#
# The file list holds one parquet path per line (blank lines and # comments ignored).
#
# Outputs of each file:
#   <model_dir>/eval/<file_stem>/
#     <ckpt_stem>__test_<file_stem>.parquet       predictions
#     <ckpt_stem>__test_<file_stem>_losses.json   loss summary
#     eval.log                                    output of the run
#   <WWW_DIR>/<model_name>/<file_stem>/           evaluation plots
# WWW_DIR defaults to /eos/user/m/mmarcheg/www/plots/ml4reco/maskformers (override through the
# environment). Once all files are done, `pb_copy_index.py -r .` is run in <WWW_DIR>/<model_name>
# to copy the web index into every plot folder.
# A failing file is reported and skipped; the script exits non-zero if any file failed.
#
# EOS is a FUSE mount that checks the Kerberos ticket of the calling process. On lxplus the ticket
# lives in $XDG_RUNTIME_DIR/krb5cc, which apptainer does not bind by default: without the binding
# every EOS path is "Permission denied" inside the container (jsonargparse then reports the config
# as "File does not exist"). A valid ticket is required (kinit); the loop renews nothing.
# The T4 GPUs of lxplus-gpu cannot run FlashAttention: the encoder attention backend is switched
# to torch (same maths, see the evaluation loop).

BASE_PATH=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")   # the hgcal-offline-reco checkout
MODEL_DIR=${1:-/eos/user/m/mmarcheg/ml4reco/maskformers/models/hgcal_v2_trackster_simtrackster_20261012-T200958}
FILE_LIST=${2:-$BASE_PATH/jobs/photon_eval_files.txt}
shift $(( $# < 2 ? $# : 2 ))
EXTRA_ARGS=("$@")

WWW_DIR=${WWW_DIR:-/eos/user/m/mmarcheg/www/plots/ml4reco/maskformers}
APPTAINER_IMAGE=${APPTAINER_IMAGE:-/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest}

MODEL_DIR=$(realpath "$MODEL_DIR")
FILE_LIST=$(realpath "$FILE_LIST")
PLOT_ROOT=$WWW_DIR/$(basename "$MODEL_DIR")
cd "$BASE_PATH" || exit 1

# ---- preflight: GPU, model, data -------------------------------------------------------------
if ! nvidia-smi; then
  echo "ERROR: no GPU visible (nvidia-smi failed). Run this on lxplus-gpu.cern.ch." >&2
  exit 1
fi
[ -f "$MODEL_DIR/config.yaml" ] || { echo "ERROR: $MODEL_DIR/config.yaml not found" >&2; exit 1; }
ls "$MODEL_DIR"/ckpts/*.ckpt >/dev/null 2>&1 || { echo "ERROR: no checkpoints in $MODEL_DIR/ckpts" >&2; exit 1; }

# Kerberos ticket for EOS inside the container: use the host's FILE: cache when there is one,
# otherwise the lxplus default $XDG_RUNTIME_DIR/krb5cc.
KRB5CC=${KRB5CCNAME#FILE:}
if [ -z "$KRB5CCNAME" ] || [ "$KRB5CC" = "$KRB5CCNAME" ]; then
  KRB5CC=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/krb5cc
fi
[ -f "$KRB5CC" ] || { echo "ERROR: Kerberos cache $KRB5CC not found; run kinit" >&2; exit 1; }
klist -s -c "FILE:$KRB5CC" || { echo "ERROR: Kerberos ticket in $KRB5CC is expired; run kinit" >&2; exit 1; }
KRB_BINDINGS=(-B "$(dirname "$KRB5CC")" --env "KRB5CCNAME=FILE:$KRB5CC")

TEST_FILES=()
MISSING=0
while read -r f; do
  if [ -f "$f" ]; then
    TEST_FILES+=("$f")
  else
    echo "ERROR: test file not found: $f" >&2
    MISSING=1
  fi
done < <(grep -v -E '^\s*(#|$)' "$FILE_LIST")
if [ "$MISSING" -ne 0 ]; then
  exit 1
fi
if ! command -v pb_copy_index.py >/dev/null; then
  echo "WARNING: pb_copy_index.py not on PATH; the plots will be written without the web index" >&2
fi

echo "Code:    $BASE_PATH"
echo "Model:   $MODEL_DIR"
echo "Files:   ${#TEST_FILES[@]} from $FILE_LIST"
echo "Plots:   $PLOT_ROOT"
echo "Image:   $APPTAINER_IMAGE"
echo "Krb5:    $KRB5CC"

# The model directory must also be readable from inside the container (EOS + Kerberos).
if ! apptainer exec -B /eos -B /afs -B /cvmfs "${KRB_BINDINGS[@]}" "$APPTAINER_IMAGE" \
     test -r "$MODEL_DIR/config.yaml"; then
  echo "ERROR: $MODEL_DIR/config.yaml is not readable inside the container (EOS/Kerberos binding)" >&2
  exit 1
fi

# ---- evaluation loop -------------------------------------------------------------------------
FAILED=()
for TEST_PATH in "${TEST_FILES[@]}"; do
  NAME=$(basename "$TEST_PATH" .parquet)
  OUT_DIR=$MODEL_DIR/eval/$NAME
  PLOT_DIR=$PLOT_ROOT/$NAME
  mkdir -p "$OUT_DIR" "$PLOT_DIR"
  echo "=================================================================================="
  echo "[$NAME] $TEST_PATH -> $OUT_DIR (plots: $PLOT_DIR)"

  # The config was written on another cluster: drop its training data_dir (the datamodule lists
  # it even when test_path is given) and point default_root_dir at the model directory.
  # lxplus-gpu has Tesla T4s (Turing): FlashAttention needs Ampere or newer, so the encoder is
  # run on the equivalent torch SDPA backend (the decoder uses it already). Later arguments win
  # in jsonargparse, so EXTRA_ARGS can still override it.
  # The prediction writer puts <ckpt_stem>__test_<test_suff>.parquet next to the checkpoint;
  # it is moved into OUT_DIR once the run (and the plot chain reading it) has finished.
  apptainer exec \
    --nv \
    --env PYTHONPATH=$BASE_PATH/src \
    -B /eos \
    -B /afs \
    -B /cvmfs \
    -B /tmp \
    "${KRB_BINDINGS[@]}" \
    "$APPTAINER_IMAGE" \
    python src/scripts/evaluate/evaluate_maskformer.py test \
      -c "$MODEL_DIR/config.yaml" \
      --trainer.default_root_dir "$MODEL_DIR" \
      --data.data_dir null \
      --data.split null \
      --data.test_path "$TEST_PATH" \
      --data.test_suff "$NAME" \
      --data.num_workers 0 \
      --trainer.precision 32-true \
      --matmul_precision highest \
      --model.model.init_args.encoder.init_args.attn_type torch \
      --plot_dir "$PLOT_DIR" \
      "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/eval.log"
  RC=${PIPESTATUS[0]}

  shopt -s nullglob
  for f in "$MODEL_DIR"/ckpts/*__test_"$NAME".parquet "$MODEL_DIR"/ckpts/*__test_"$NAME"_losses.json; do
    mv "$f" "$OUT_DIR/"
  done
  shopt -u nullglob

  if [ "$RC" -ne 0 ]; then
    echo "[$NAME] FAILED with exit code $RC (log: $OUT_DIR/eval.log)"
    FAILED+=("$NAME")
  else
    echo "[$NAME] done -> $OUT_DIR (plots: $PLOT_DIR)"
  fi
done

echo "=================================================================================="
if command -v pb_copy_index.py >/dev/null; then
  echo "Copying the web index into $PLOT_ROOT"
  (cd "$PLOT_ROOT" && pb_copy_index.py -r .) || echo "WARNING: pb_copy_index.py failed"
else
  echo "WARNING: pb_copy_index.py not on PATH; run it by hand: cd $PLOT_ROOT && pb_copy_index.py -r ."
fi

echo "=================================================================================="
echo "Evaluated $(( ${#TEST_FILES[@]} - ${#FAILED[@]} ))/${#TEST_FILES[@]} file(s); outputs in $MODEL_DIR/eval/, plots in $PLOT_ROOT"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Failed: ${FAILED[*]}"
  exit 1
fi
