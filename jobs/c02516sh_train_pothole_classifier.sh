#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pothole_classifier_l40s
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o outputs/pothole_classifier_c02516_%J.out
#BSUB -e outputs/pothole_classifier_c02516_%J.err

cd ~/projects/object-detection
mkdir -p outputs

uv sync

uv run python scripts/train_pothole_classifier.py