import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
import cv2
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image

from src.models.pothole_classifier import PotholeClassifier
from src.datasets.parse_xml import parse_pothole_xml
from src.utils.nms import nms
from src.utils.iou import compute_iou
from src.utils.checkpoints import load_checkpoint
from src.utils.logger import get_logger
from src.utils.transforms import get_val_transforms


# Paths
DATASET_PATH = Path("/dtu/datasets1/02516/potholes")
PROPOSALS_PATH = "src/datasets/proposals/proposals_test.pkl"
CHECKPOINT_PATH = "checkpoints/20251206_195414/best_model.pth" #NOTE: Update this to the trained model path to evaluate
SPLITS_PATH = "splits.json"
RESULTS_DIR = "results/"

# Model Configuration (must match training)
NUM_CLASSES = 2
DROPOUT_P = 0.5

# Detection Hyperparameters
SCORE_THRESHOLD = 0.0 # Confidence threshold for initial filtering
NMS_IOU_THRESHOLD = 0.5 # IoU threshold for NMS
DETECTION_IOU_THRESHOLDS = [0.1, 0.3, 0.5, 0.7] # IoU thresholds for mAP calculation

# Data Configuration
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logger = get_logger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_image_and_proposals(img_name, dataset_path, all_proposals):
    """Load full image, ground truth boxes, and proposals for a single image."""
    # Load image
    img_path = dataset_path / "images" / img_name
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load ground truth from XML
    xml_name = img_name.replace('.jpg', '.xml').replace('.png', '.xml')
    xml_path = dataset_path / "annotations" / xml_name
    gt_boxes = parse_pothole_xml(str(xml_path))
    
    # Load proposals for this image
    proposals = all_proposals.get(img_name, [])
    
    return image, gt_boxes, proposals


def classify_proposals(model, image, proposals, transform, device):
    """Classify all proposals for a single image."""
    if len(proposals) == 0:
        logger.warning("No proposals found for this image.")
        return np.array([])
    
    model.eval()
    scores = []
    
    with torch.no_grad():
        for box in proposals:
            x1, y1, x2, y2 = map(int, box)
            
            # Clamp coordinates to image boundaries
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                scores.append(0.0)
                continue
            
            # Crop proposal
            patch = image[y1:y2, x1:x2]
            
            # Handle empty patches
            if patch.size == 0:
                scores.append(0.0)
                continue
            
            # Transform
            patch_pil = Image.fromarray(patch)
            patch_tensor = transform(patch_pil).unsqueeze(0).to(device)
            
            # Predict
            output = model(patch_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].item()  # Probability of class 1 (pothole)
            scores.append(prob)
    
    return np.array(scores)


def apply_nms_to_detections(boxes, scores, nms_iou_threshold):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    if len(boxes) == 0:
        return np.array([]), np.array([])
    
    # Apply NMS
    keep_indices = nms(boxes, scores, iou_thresh=nms_iou_threshold, score_thresh=0.0)
    
    return boxes[keep_indices], scores[keep_indices]


def match_detections_to_ground_truth(detections, det_scores, gt_boxes, iou_threshold):
    """Match detections to ground truth boxes using IoU threshold."""
    if len(detections) == 0:
        return [], [], [], set()
    
    if len(gt_boxes) == 0:
        # All detections are false positives
        return [], [], list(range(len(detections))), set()
    
    # Convert gt_boxes to numpy array
    gt_boxes_array = np.array(gt_boxes)
    
    matches = []
    matched_gt = set()
    
    # For each detection, find best matching ground truth
    for det_idx, det_box in enumerate(detections):
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes_array):
            if gt_idx in matched_gt:
                continue  # This GT already matched
            
            iou_val = compute_iou(det_box, gt_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            matches.append((det_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
    
    # Classify detections as TP or FP
    tp_indices = [m[0] for m in matches]
    fp_indices = [i for i in range(len(detections)) if i not in tp_indices]
    
    return matches, tp_indices, fp_indices, matched_gt


def calculate_precision_recall(all_detections, all_ground_truths, iou_threshold):
    """Calculate precision-recall curve for all images."""
    # Collect all detections across images with their scores
    all_det_scores = []
    all_det_matched = []  # True if TP, False if FP
    
    total_gt = sum(len(gt) for gt in all_ground_truths)
    
    for (det_boxes, det_scores), gt_boxes in zip(all_detections, all_ground_truths):
        if len(det_boxes) == 0:
            continue
        
        matches, tp_indices, fp_indices, matched_gt = match_detections_to_ground_truth(
            det_boxes, det_scores, gt_boxes, iou_threshold
        )
        
        # Store each detection with its score and TP/FP label
        for i, score in enumerate(det_scores):
            all_det_scores.append(score)
            all_det_matched.append(i in tp_indices)
    
    if len(all_det_scores) == 0:
        return [0], [0], 0.0
    
    # Sort detections by score (descending)
    sorted_indices = np.argsort(all_det_scores)[::-1]
    all_det_matched = np.array(all_det_matched)[sorted_indices]
    
    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(all_det_matched)
    fp_cumsum = np.cumsum(~all_det_matched)
    
    # Calculate precision and recall at each threshold
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gt if total_gt > 0 else tp_cumsum
    
    # Add endpoints
    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])
    
    # Calculate AP using all-point interpolation
    ap = 0.0
    for i in range(len(recalls) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
    
    return precisions.tolist(), recalls.tolist(), float(ap)


def calculate_map(all_detections, all_ground_truths, iou_thresholds):
    """Calculate mAP for multiple IoU thresholds."""
    results = {}
    
    for iou_thresh in iou_thresholds:
        precisions, recalls, ap = calculate_precision_recall(
            all_detections, all_ground_truths, iou_thresh
        )
        results[iou_thresh] = {
            'ap': ap,
            'precisions': precisions,
            'recalls': recalls
        }
    
    return results


def save_results(results, results_dir, filename=f"{timestamp}_detection_results.json"):
    """Save detection evaluation results to JSON file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = results_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
    return filepath


def main():
    """Main detection evaluation function."""
    
    logger.info("Pothole Detection System")
    
    # Log device info
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    
    # Load test split
    logger.info(f"\nLoading test split from: {SPLITS_PATH}")
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)
    
    test_images = splits['test']
    logger.info(f"Test images: {len(test_images)}")
    
    # Load all proposals from pickle file
    logger.info(f"\nLoading proposals from: {PROPOSALS_PATH}")
    with open(PROPOSALS_PATH, 'rb') as f:
        all_proposals = pickle.load(f)
    logger.info(f"Proposals loaded for {len(all_proposals)} images")
    
    # Load trained model
    logger.info(f"\nLoading model from: {CHECKPOINT_PATH}")
    model = PotholeClassifier(
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,
        dropout_p=DROPOUT_P
    )
    
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    epoch, best_map = load_checkpoint(checkpoint_path, model, device=DEVICE)
    logger.info(f"Loaded model from epoch {epoch}")
    if best_map is not None:
        logger.info(f"Model's validation mAP (classification): {best_map:.4f}")
    
    model = model.to(DEVICE)
    model.eval()
    
    # Create transform
    transform = get_val_transforms(img_size=IMG_SIZE)
    
    # Process each test image    
    logger.info("Processing test images...")
    
    all_detections = []
    all_ground_truths = []
    total_proposals = 0
    total_detections_before_nms = 0
    total_detections_after_nms = 0
    
    for img_name in tqdm(test_images, desc="Evaluating"):
        # Load image, ground truth, and proposals
        image, gt_boxes, proposals = load_image_and_proposals(
            img_name, DATASET_PATH, all_proposals
        )
        
        total_proposals += len(proposals)
        
        if len(proposals) == 0:
            all_detections.append((np.array([]), np.array([])))
            all_ground_truths.append(gt_boxes)
            continue
        
        # Classify all proposals
        scores = classify_proposals(model, image, proposals, transform, DEVICE)
        
        # Filter by score threshold
        proposals_array = np.array(proposals)
        keep = scores >= SCORE_THRESHOLD
        filtered_boxes = proposals_array[keep]
        filtered_scores = scores[keep]
        
        total_detections_before_nms += len(filtered_boxes)
        
        # Apply NMS
        final_boxes, final_scores = apply_nms_to_detections(
            filtered_boxes, filtered_scores, NMS_IOU_THRESHOLD
        )
        
        total_detections_after_nms += len(final_boxes)
        
        # Store results
        all_detections.append((final_boxes, final_scores))
        all_ground_truths.append(gt_boxes)
    
    # Calculate detection mAP
    logger.info("Calculating detection mAP...")
    
    map_results = calculate_map(all_detections, all_ground_truths, DETECTION_IOU_THRESHOLDS)
    
    # Log summary
    total_gt = sum(len(gt) for gt in all_ground_truths)
    
    logger.info("DETECTION EVALUATION RESULTS")
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Test images:                    {len(test_images):>6,}")
    logger.info(f"  Ground truth potholes:          {total_gt:>6,}")
    logger.info(f"  Total proposals:                {total_proposals:>6,}")
    logger.info(f"  Proposals per image (avg):      {total_proposals/len(test_images):>6.1f}")
    
    logger.info(f"\nDetection Pipeline:")
    logger.info(f"  Score threshold:                {SCORE_THRESHOLD:>6.2f}")
    logger.info(f"  Detections before NMS:          {total_detections_before_nms:>6,}")
    logger.info(f"  NMS IoU threshold:              {NMS_IOU_THRESHOLD:>6.2f}")
    logger.info(f"  Detections after NMS:           {total_detections_after_nms:>6,}")
    logger.info(f"  Final detections per image:     {total_detections_after_nms/len(test_images):>6.1f}")
    
    logger.info(f"\nDetection Performance:")
    for iou_thresh in DETECTION_IOU_THRESHOLDS:
        ap = map_results[iou_thresh]['ap']
        logger.info(f"  mAP@{iou_thresh:.2f}:                       {ap:>6.2%}")
        
    # Save results    
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(CHECKPOINT_PATH),
        "test_set": str(DATASET_PATH),
        "num_test_images": len(test_images),
        "hyperparameters": {
            "score_threshold": SCORE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "detection_iou_thresholds": DETECTION_IOU_THRESHOLDS,
        },
        "statistics": {
            "total_ground_truth": total_gt,
            "total_proposals": total_proposals,
            "avg_proposals_per_image": total_proposals / len(test_images),
            "detections_before_nms": total_detections_before_nms,
            "detections_after_nms": total_detections_after_nms,
            "avg_detections_per_image": total_detections_after_nms / len(test_images),
        },
        "detection_metrics": {
            f"map_{iou_thresh}": map_results[iou_thresh]['ap']
            for iou_thresh in DETECTION_IOU_THRESHOLDS
        }
    }
    
    results_file = save_results(results, RESULTS_DIR)
    logger.info(f"Detection evaluation completed. Results saved to {results_file}")
    
if __name__ == "__main__":
    main()