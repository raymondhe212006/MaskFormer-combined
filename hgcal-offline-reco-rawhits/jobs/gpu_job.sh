#!/bin/bash
#SBATCH --job-name=hgcal_maskformer_gpu
#SBATCH --output=logs/slurm/hgcal_maskformer_gpu_%j.out
#SBATCH --partition=work
#SBATCH --gres=mps:50              # Request 50% of the GPU (max 50%)
#SBATCH --cpus-per-task=32         # Match CPU threads to GPU work
#SBATCH --mem=128G                  # Max 4G per CPU (32 CPUs x 4G = 128G limit)

BASE_PATH=/home/export/$USER/ml4reco/hgcal-offline-reco

nvidia-smi
echo "CUDA_VISIBLE_DEVICES on host: ${CUDA_VISIBLE_DEVICES:-<not set>}"

cd $BASE_PATH

# Launch the training script inside the Apptainer container with GPU support
APPTAINER_IMAGE=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest
echo "Starting training MaskFormer on GPU within Apptainer container..."
echo "Using Apptainer image: $APPTAINER_IMAGE" 

# With --gres=mps:X allocations SLURM may not set CUDA_VISIBLE_DEVICES; fall back to GPU 0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

apptainer exec \
  --nv \
  --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  --env LD_LIBRARY_PATH=/host-nvidia-libs:$LD_LIBRARY_PATH \
  -B /home/export/$USER \
  -B /cvmfs \
  -B /tmp \
  -B /usr/lib/x86_64-linux-gnu:/host-nvidia-libs \
  $APPTAINER_IMAGE \
  bash -c "
    echo \"CUDA_VISIBLE_DEVICES inside container: \$CUDA_VISIBLE_DEVICES\"
    echo \"LD_LIBRARY_PATH: \$LD_LIBRARY_PATH\"
    nvidia-smi
    export PYTHONPATH=$BASE_PATH/src:\$PYTHONPATH
    python src/scripts/train/train_maskformer.py fit \
      --config configs/base.yaml \
      --data.num_workers 0 \
      --trainer.num_sanity_val_steps 0
  "
