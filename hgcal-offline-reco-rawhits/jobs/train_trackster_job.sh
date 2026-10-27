#!/bin/bash
#SBATCH --job-name=hgcal_trackster_simtrackster
#SBATCH --output=logs/slurm/hgcal_trackster_simtrackster_%j.out
#SBATCH --partition=work
#SBATCH --qos=light                # 7 days, max mps:25 per job (medium allows mps:50 but only 8h)
#SBATCH --time=7-00:00:00
#SBATCH --gres=mps:25              # 25% of the GPU, the light qos per-job limit
#SBATCH --cpus-per-task=32         # Match CPU threads to GPU work
#SBATCH --mem=128G                 # Max 4G per CPU (32 CPUs x 4G = 128G limit)

# Trains configs/base_trackster_simtrackster.yaml on a directory of parquet files, split
# 80/10/10 into train/val/test by whole files (see HGCDataModule data_dir/split).
# Usage: sbatch jobs/train_trackster_job.sh [data_dir]

BASE_PATH=/home/export/$USER/ml4reco/hgcal-offline-reco
DATA_DIR=${1:-/home/export/mmarcheg/store/ml4reco/data_maskformers/taus_noFineCalo_with_tracksters_50k}

nvidia-smi
echo "CUDA_VISIBLE_DEVICES on host: ${CUDA_VISIBLE_DEVICES:-<not set>}"

cd $BASE_PATH

APPTAINER_IMAGE=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest
echo "Starting training MaskFormer on GPU within Apptainer container..."
echo "Using Apptainer image: $APPTAINER_IMAGE"
echo "Using data directory: $DATA_DIR"

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
    nvidia-smi
    export PYTHONPATH=$BASE_PATH/src:\$PYTHONPATH
    python src/scripts/train/train_maskformer.py fit \
      --config configs/base_trackster_simtrackster.yaml \
      --data.train_path null \
      --data.valid_path null \
      --data.test_path null \
      --data.data_dir $DATA_DIR \
      --data.split '[0.8,0.1,0.1]' \
      --data.split_seed 42 \
      --data.num_workers 0 \
      --trainer.num_sanity_val_steps 0
  "
