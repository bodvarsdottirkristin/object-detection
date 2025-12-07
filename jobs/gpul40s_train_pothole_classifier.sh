#!/bin/sh
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pothole_classifier_l40s
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 12:00
#BSUB -o outputs/pothole_classifier_l40s_%J.out
#BSUB -e outputs/pothole_classifier_l40s_%J.err

cd ~/projects/object-detection
mkdir -p outputs

uv sync

uv run python scripts/train_pothole_classifier.py