
# Pothole Object Detection

**Note:** This project uses [uv](https://github.com/astral-sh/uv) for Python package management and execution. Before running any scripts, install dependencies with:

```sh
uv sync
```

All commands below should be run using `uv run -m ...`.

This project detects potholes in road images using deep learning and object detection techniques. It provides a full pipeline from dataset preparation to model training and evaluation.

## Overview

- Uses the Potholes dataset (images + PascalVOC XML annotations)
- Generates object proposals with Selective Search
- Labels proposals for training a classifier
- Trains and evaluates a deep learning classifier to detect potholes

## Pipeline: Step-by-Step

1. **Create Dataset Splits**
   - Generates train/val/test splits from images with matching XML annotations.
   - Run:

    ```sh
    uv run -m src.datasets.create_splits
    ```

2. **Extract Object Proposals**
   - Runs Selective Search to generate bounding box proposals for each image.
   - Run:

    ```sh
    uv run -m src.datasets.proposals.create_proposals
    ```

   - Outputs: `proposals_train.pkl`, `proposals_val.pkl`, `proposals_test.pkl`

3. **Label Proposals**
   - Assigns pothole/background labels to each proposal using IoU with ground truth.
   - Run:

    ```sh
    uv run -m src.datasets.proposals.label_proposals
    ```

   - Outputs: `train_db.pkl`, `val_db.pkl`, `test_db.pkl` (for classifier training)

4. **Train Pothole Classifier**
   - Trains a ResNet-based classifier on cropped proposal patches.
   - Run:

    ```sh
    uv run -m scripts.train_pothole_classifier
    ```

   - Model: See `src/models/pothole_classifier.py`
   - Dataset: See `src/datasets/potholes.py`
   - Training logic: See `src/utils/train.py`, `src/utils/losses.py`, `src/utils/checkpoints.py`

5. **Evaluate Classifier**
   - Evaluates classifier performance on test proposals.
   - Run:

    ```sh
    uv run -m scripts.evaluate_pothole_classifier
    ```

   - Metrics: Accuracy, AUC, mAP, Precision, Recall, F1, etc. (see `src/utils/evaluate.py`)

6. **Evaluate Detector**
   - Runs full detection pipeline: classifies proposals, applies NMS, matches detections to ground truth.
   - Run:

    ```sh
    uv run -m scripts.evaluate_pothole_detector
    ```

   - Uses: `src/utils/nms.py`, `src/utils/iou.py`, `src/datasets/parse_xml.py`

## Notes

- Update dataset paths in scripts as needed.
- All outputs and checkpoints are saved in relevant folders (`src/datasets/proposals/`, `checkpoints/`, `results/`).
- For details on each module, see the corresponding Python files.
