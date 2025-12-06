#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pothole_step1_inference
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 02:00
#BSUB -o outputs/step1_inference_%J.out
#BSUB -e outputs/step1_inference_%J.err

mkdir -p outputs

module load python3/3.13.9

# Navigate to project and sync
cd ~/project/object-detection
uv sync

# Run Step 1: Apply CNN to test proposals
echo "========================================"
echo "Step 1: Apply CNN to Test Proposals"
echo "========================================"
uv run python test.py
