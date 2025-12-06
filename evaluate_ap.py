"""Step 3: Evaluate object detection using Average Precision (AP) metric."""
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from IoU import iou


def compute_ap(recall, precision):
    """
    Compute Average Precision from recall and precision curves.
    
    Uses the 11-point interpolation method (PASCAL VOC style).
    """
    # Append sentinel values at the beginning and end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute the precision envelope (monotonically decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate area under the curve (AP)
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def evaluate_detections(predictions_file, proposals_file, ground_truth_file, iou_threshold=0.5):
    """
    Evaluate object detection predictions against ground truth.
    
    Args:
        predictions_file: Path to predictions from Step 2 (NMS results)
        proposals_file: Path to test proposals (proposals_test.pkl)
        ground_truth_file: Path to ground truth labels (labels_test.pkl)
        iou_threshold: IoU threshold for matching (default 0.5)
        
    Returns:
        dict: AP score and detailed metrics
    """
    
    print(f"\n{'='*70}")
    print("STEP 3: EVALUATE OBJECT DETECTION (AVERAGE PRECISION)")
    print(f"{'='*70}\n")
    
    # Load predictions (after NMS)
    print("Loading NMS predictions...")
    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)
    print(f"✓ Loaded {len(predictions)} images\n")
    
    # Load proposals
    print("Loading test proposals...")
    with open(proposals_file, 'rb') as f:
        proposals_dict = pickle.load(f)
    print(f"✓ Loaded proposals\n")
    
    # Load ground truth labels (binary labels for each proposal)
    print("Loading ground truth labels...")
    with open(ground_truth_file, 'rb') as f:
        ground_truth_labels = pickle.load(f)
    print(f"✓ Loaded ground truth for {len(ground_truth_labels)} images\n")
    
    # Collect all detections and ground truth across all images
    all_detections = []  # (image_id, confidence, bbox)
    all_ground_truth = []  # (image_id, bbox)
    
    num_images = 0
    total_gt = 0
    total_detections = 0
    
    print("Processing detections and ground truth...")
    for image_filename in tqdm(predictions.keys(), desc="Processing"):
        # Get predictions for this image (after NMS)
        pred = predictions[image_filename]
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        
        # Get proposals for this image
        if image_filename not in proposals_dict:
            continue
        all_proposals = proposals_dict[image_filename]['proposals']
        
        # Get ground truth labels for this image
        if image_filename not in ground_truth_labels:
            continue
        gt_labels = ground_truth_labels[image_filename]
        
        num_images += 1
        total_detections += len(pred_boxes)
        
        # Add predictions (these are already filtered pothole detections)
        for box, score in zip(pred_boxes, pred_scores):
            all_detections.append({
                'image_id': image_filename,
                'confidence': float(score),
                'bbox': box
            })
        
        # Get ground truth boxes (proposals labeled as potholes)
        gt_indices = np.where(np.array(gt_labels) == 1)[0]
        gt_boxes = np.array(all_proposals)[gt_indices]
        
        total_gt += len(gt_boxes)
        
        # Add ground truth
        for box in gt_boxes:
            all_ground_truth.append({
                'image_id': image_filename,
                'bbox': box
            })
    
    print(f"✓ Processing complete\n")
    
    print("="*70)
    print("Detection Statistics:")
    print("="*70)
    print(f"  Images evaluated:       {num_images}")
    print(f"  Total ground truth:     {total_gt}")
    print(f"  Total detections:       {total_detections}\n")
    
    if total_gt == 0:
        print("ERROR: No ground truth found!")
        return 0, None
    
    # Sort detections by confidence (descending)
    all_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
    
    # Match detections to ground truth
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    
    # Track which ground truth boxes have been matched
    gt_matched = {}
    for gt in all_ground_truth:
        gt_id = (gt['image_id'], tuple(gt['bbox']))
        gt_matched[gt_id] = False
    
    print("Matching detections to ground truth...")
    for det_idx, detection in tqdm(enumerate(all_detections), total=len(all_detections), desc="Matching"):
        det_box = detection['bbox']
        det_image = detection['image_id']
        
        # Find ground truth boxes for this image
        image_gts = [gt for gt in all_ground_truth if gt['image_id'] == det_image]
        
        if len(image_gts) == 0:
            # No ground truth for this image, so this is a false positive
            fp[det_idx] = 1
            continue
        
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(image_gts):
            gt_box = gt['bbox']
            current_iou = iou(det_box, gt_box)
            
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = gt_idx
        
        # Check if best match exceeds IoU threshold
        if best_iou >= iou_threshold:
            gt_id = (det_image, tuple(image_gts[best_gt_idx]['bbox']))
            
            if not gt_matched[gt_id]:
                # True positive
                tp[det_idx] = 1
                gt_matched[gt_id] = True
            else:
                # False positive (already matched)
                fp[det_idx] = 1
        else:
            # False positive (IoU below threshold)
            fp[det_idx] = 1
    
    print(f"✓ Matching complete\n")
    
    # Compute cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute recall and precision
    recalls = tp_cumsum / total_gt if total_gt > 0 else np.zeros(len(tp))
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Compute AP
    ap = compute_ap(recalls, precisions)
    
    print("="*70)
    print("AVERAGE PRECISION (AP) RESULTS")
    print("="*70)
    print(f"  IoU threshold:          {iou_threshold}")
    print(f"  Average Precision (AP): {ap:.4f}")
    print(f"  True Positives:         {int(tp_cumsum[-1])}")
    print(f"  False Positives:        {int(fp_cumsum[-1])}")
    print(f"  Max Recall:             {recalls[-1]:.4f}")
    print(f"  Max Precision:          {precisions[0]:.4f}\n")
    
    # Compute additional metrics at different recall levels
    print("Precision at different recall levels:")
    for recall_level in [0.1, 0.25, 0.5, 0.75, 0.9]:
        idx = np.where(recalls >= recall_level)[0]
        if len(idx) > 0:
            max_precision = np.max(precisions[idx])
            print(f"  @ Recall {recall_level:.2f}: {max_precision:.4f}")
    print()
    
    # Save results
    results = {
        'ap': float(ap),
        'iou_threshold': iou_threshold,
        'num_images': num_images,
        'total_gt': total_gt,
        'total_detections': total_detections,
        'tp': int(tp_cumsum[-1]),
        'fp': int(fp_cumsum[-1]),
        'recalls': recalls.tolist(),
        'precisions': precisions.tolist(),
        'max_recall': float(recalls[-1]),
        'max_precision': float(precisions[0])
    }
    
    output_file = Path(predictions_file).parent / 'evaluation_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to: {output_file}\n")
    
    return ap, results


def plot_pr_curve(results, output_path='pr_curve.png'):
    """
    Plot Precision-Recall curve (optional, requires matplotlib).
    """
    try:
        import matplotlib.pyplot as plt
        
        recalls = results['recalls']
        precisions = results['precisions']
        ap = results['ap']
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, 'b-', linewidth=2, label=f'AP = {ap:.4f}')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ PR curve saved to: {output_path}\n")
        plt.close()
    except ImportError:
        print("(Matplotlib not available, skipping PR curve plot)\n")


if __name__ == "__main__":
    # Configuration
    PREDICTIONS_FILE = "scratch/proposals/test_predictions_nms.pkl"
    PROPOSALS_FILE = "scratch/proposals/proposals_test.pkl"
    GROUND_TRUTH_FILE = "scratch/proposals/labels_test.pkl"
    IOU_THRESHOLD = 0.5
    
    # Evaluate
    ap, results = evaluate_detections(
        PREDICTIONS_FILE,
        PROPOSALS_FILE,
        GROUND_TRUTH_FILE,
        iou_threshold=IOU_THRESHOLD
    )
    
    # Plot PR curve (optional)
    if results:
        plot_pr_curve(results, output_path='scratch/proposals/pr_curve.png')
    
    print("="*70)
    print("✓ Step 3 complete!")
    print("="*70)
    print("\nAll steps finished! Check results above.")