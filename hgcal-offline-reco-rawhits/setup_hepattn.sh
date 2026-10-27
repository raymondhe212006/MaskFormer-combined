INITIAL_DIR=$(pwd)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEPATTN_DIR=${HEPATTN_DIR:-~/ml4reco/hepattn}

# Activate environment with hepattn and flash_attn installed
export APPTAINERENV_PYTHONPATH=$BASE_DIR/src:$PYTHONPATH
cd $HEPATTN_DIR
apptainer exec --nv pixi.sif bash -c "pixi shell -e default"
cd $INITIAL_DIR
