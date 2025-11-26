import pickle
import json
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.datasets.parse_xml import parse_pothole_xml

# --- Configuration ---
DATASET_PATH = Path("/dtu/datasets1/02516/potholes/")
PROPOSALS_PATH = Path("src/datasets/proposals/proposals_train.pkl")
IOU_THRESHOLD = 0.5


def _compute_iou_matrix(proposals, gt_boxes):
    """
    Computes IoU between every proposal and every GT box.
    Returns matrix of shape (num_proposals, num_gt)
    """
    if len(proposals) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(proposals), len(gt_boxes)))

    proposals = np.array(proposals)
    gt_boxes = np.array(gt_boxes)

    # Coordinates
    xA = np.maximum(proposals[:, 0:1], gt_boxes[:, 0:1].T)
    yA = np.maximum(proposals[:, 1:2], gt_boxes[:, 1:2].T)
    xB = np.minimum(proposals[:, 2:3], gt_boxes[:, 2:3].T)
    yB = np.minimum(proposals[:, 3:4], gt_boxes[:, 3:4].T)

    # Intersection
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # Unions
    boxAArea = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    boxBArea = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    
    # Broadcast areas to match matrix shape
    boxAArea = boxAArea[:, np.newaxis]
    boxBArea = boxBArea[np.newaxis, :]
    
    iou_matrix = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou_matrix

def main():
    # 1. Load Proposals
    print(f"Loading proposals from {PROPOSALS_PATH}...")
    with open(PROPOSALS_PATH, 'rb') as f:
        all_proposals = pickle.load(f)

    # 2. Define thresholds to test (e.g., Top 10, 50, 100 ... 2000)
    k_values = [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]
    total_gt_objects = 0
    # Store how many GTs are detected at each k level
    detected_counts = {k: 0 for k in k_values}

    print("Evaluating Recall...")
    for img_name, proposals in tqdm(all_proposals.items()):
        # Load GT
        xml_path = DATASET_PATH / 'annotations' / img_name.replace('.png', '.xml')
        gt_boxes = parse_pothole_xml(xml_path)
        
        num_gt = len(gt_boxes)
        if num_gt == 0: continue
        total_gt_objects += num_gt

        # Calculate IoU matrix
        # Matrix shape: [Num_Proposals, Num_GT]
        iou_matrix = _compute_iou_matrix(proposals, gt_boxes)

        # Check recall at each k
        for k in k_values:
            # Take top k proposals
            current_ious = iou_matrix[:k] 
            if current_ious.size == 0: continue
            
            # For each GT column, is there ANY proposal with IoU > 0.5?
            max_ious_per_gt = np.max(current_ious, axis=0)
            detected_gt = np.sum(max_ious_per_gt >= IOU_THRESHOLD)
            detected_counts[k] += detected_gt

    # 3. Calculate and Plot Results
    recalls = [detected_counts[k] / total_gt_objects for k in k_values]
    
    print("\n--- Results ---")
    print(f"Total Ground Truth Potholes: {total_gt_objects}")
    for k, rec in zip(k_values, recalls):
        print(f"Top {k} proposals: Recall = {rec:.4f} ({rec*100:.2f}%)")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, recalls, marker='o', linewidth=2)
    plt.title(f'Recall vs Number of Proposals (IoU >= {IOU_THRESHOLD})')
    plt.xlabel('Number of Proposals (k)')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Target')
    plt.legend()
    plt.savefig('recall_plot.png')
    print("\n✓ Plot saved to recall_plot.png")

if __name__ == "__main__":
    main()