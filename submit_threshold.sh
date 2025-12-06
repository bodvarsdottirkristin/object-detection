#!/bin/bash
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pothole_threshold
#BSUB -W 04:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o outputs/threshold_%J.out
#BSUB -e outputs/threshold_%J.err

uv run python optimal_threshold.py