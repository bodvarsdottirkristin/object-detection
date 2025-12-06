"""Step 2: Apply NMS to test predictions from Step 1."""
import sys
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import from scripts folder
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from NMS import nms


def apply_nms_to_predictions(predictions_file, output_file, iou_threshold=0.5, score_threshold=0.5):
    """
    Apply NMS to predictions from Step 1 for all test images.
    
    Args:
        predictions_file: Path to test_predictions.pkl from Step 1
        output_file: Path to save NMS results
        iou_threshold: IoU threshold for NMS
        score_threshold: Confidence threshold for filtering boxes
    """
    
    print(f"\n{'='*70}")
    print("STEP 2: APPLY NMS (NON-MAXIMUM SUPPRESSION)")
    print(f"{'='*70}\n")
    
    # Load predictions
    print(f"Loading predictions from: {predictions_file}")
    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)
    print(f"✓ Loaded predictions for {len(predictions)} images\n")
    
    # Apply NMS
    print(f"Applying NMS (IoU threshold: {iou_threshold}, Score threshold: {score_threshold})...")
    nms_results = {}
    
    total_before = 0
    total_after = 0
    
    for image_filename in tqdm(predictions.keys(), desc="NMS Progress"):
        pred = predictions[image_filename]
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']
        
        if len(boxes) == 0:
            nms_results[image_filename] = {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }
            continue
        
        # Apply NMS (returns indices of boxes to keep)
        keep_indices = nms(boxes, scores, iou_thresh=iou_threshold, score_thresh=score_threshold)
        
        nms_results[image_filename] = {
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'labels': labels[keep_indices]
        }
        
        total_before += len(boxes)
        total_after += len(keep_indices)
    
    print(f"✓ NMS complete\n")
    
    # Print statistics
    print("="*70)
    print("NMS Statistics:")
    print("="*70)
    print(f"  Total proposals before NMS: {total_before}")
    print(f"  Total detections after NMS:  {total_after}")
    if total_before > 0:
        reduction = 100 * (total_before - total_after) / total_before
        print(f"  Boxes removed:              {total_before - total_after} ({reduction:.1f}%)\n")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(nms_results, f)
    print(f"✓ NMS results saved to: {output_file}\n")
    
    return nms_results


if __name__ == "__main__":
    # Configuration
    PREDICTIONS_FILE = "scratch/proposals/test_predictions.pkl"
    OUTPUT_FILE = "scratch/proposals/test_predictions_nms.pkl"
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.5
    
    # Apply NMS
    nms_results = apply_nms_to_predictions(
        PREDICTIONS_FILE,
        OUTPUT_FILE,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=SCORE_THRESHOLD
    )
    
    print("="*70)
    print("✓ Step 2 complete!")
    print("="*70)
    print("\nNext: Implement Average Precision (AP) evaluation in Step 3")