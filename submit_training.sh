#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pothole_training
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o outputs/output_pothole_%J.out
#BSUB -e outputs/output_pothole_%J.err

mkdir -p outputs

module load python3/3.13.9

# Navigate to project and sync
cd ~/project/object-detection
uv sync

# Set training parameters
export NUM_EPOCHS=50
export BATCH_SIZE=32

# Run training from src/training
uv run python train.py