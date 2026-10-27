#!/usr/bin/env bash
# Run the standard MaskFormer plotting chain on a prediction file by hand
# (src/scripts/evaluate/evaluate_maskformer.py runs the same chain automatically after `test`).
#
#   ./src/scripts/plot/run_all.sh <prediction.parquet> [outdir] [extra run_plot_chain.py args...]
#
# The (topocluster, particle) collections are auto-detected from the file.
# Needs awkward + numpy + pyarrow + matplotlib (mplhep optional); interpreter: $PYTHON or `python`.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <prediction.parquet> [outdir] [extra run_plot_chain.py args...]" >&2
    exit 2
fi
INPUT="$1"; shift
OUTDIR="plots/maskformer_eval"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    OUTDIR="$1"; shift
fi
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTDIR"

echo "### input  : $INPUT"
echo "### outdir : $OUTDIR"
echo "### python : $PY"

"$PY" "$HERE/run_plot_chain.py" --input "$INPUT" --outdir "$OUTDIR" "$@" 2>&1 | tee "$OUTDIR/plot_chain.log"
exit "${PIPESTATUS[0]}"
