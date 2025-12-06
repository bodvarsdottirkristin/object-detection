"""Find optimal pothole confidence threshold using validation data."""
import torch
import pickle
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from IoU import iou

from src.models.pothole_ResNet import PotholeClassifier


def get_all_confidences(model_path, proposals_dir, dataset_path, device, split='val'):
    """
    Get all pothole confidence scores for a dataset split.
    
    Returns:
        - confidence_scores: array of all pothole confidence scores
        - all_data: detailed info for threshold analysis
    """
    
    print(f"\nAnalyzing {split} set to find optimal threshold...")
    
    # Load model
    model = PotholeClassifier(num_classes=2, pretrained=True).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load proposals and labels
    proposals_file = Path(proposals_dir) / f'proposals_{split}.pkl'
    labels_file = Path(proposals_dir) / f'labels_{split}.pkl'
    
    with open(proposals_file, 'rb') as f:
        proposals_dict = pickle.load(f)
    
    with open(labels_file, 'rb') as f:
        labels_dict = pickle.load(f)
    
    # Normalization stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    all_data = []
    all_confidences = []
    
    with torch.no_grad():
        for image_filename in tqdm(proposals_dict.keys(), desc=f"Getting {split} confidences"):
            proposals = proposals_dict[image_filename]['proposals']
            gt_labels = labels_dict[image_filename]
            
            if len(proposals) == 0:
                continue
            
            # Load image
            image_path = Path(dataset_path) / 'images' / image_filename
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract patches
            patches = []
            for box in proposals:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                patch = image[y1:y2, x1:x2]
                if patch.size == 0:
                    patch = np.zeros((64, 64, 3), dtype=np.uint8)
                else:
                    patch = cv2.resize(patch, (224, 224))
                
                patch = patch.astype(np.float32) / 255.0
                patch = (patch - mean) / std
                patch = torch.tensor(patch).permute(2, 0, 1).float()
                patches.append(patch)
            
            # Get predictions
            patches_tensor = torch.stack(patches)
            batch_size = 16
            all_scores = []
            all_labels = []
            
            for i in range(0, len(patches_tensor), batch_size):
                batch = patches_tensor[i:i+batch_size].to(device)
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=1)
                scores = probabilities[:, 1]
                labels = torch.argmax(logits, dim=1)
                
                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            scores_array = np.concatenate(all_scores, axis=0)
            labels_array = np.concatenate(all_labels, axis=0)
            
            # Store data
            for idx, (prop, score, label, gt_label) in enumerate(
                zip(proposals, scores_array, labels_array, gt_labels)
            ):
                all_data.append({
                    'image': image_filename,
                    'box': prop,
                    'pothole_confidence': score,
                    'predicted_label': label,
                    'gt_label': gt_label,
                    'is_correct': (label == gt_label)
                })
                all_confidences.append(score)
    
    return np.array(all_confidences), all_data


def analyze_thresholds(all_data):
    """
    Analyze different thresholds to find optimal one.
    """
    
    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS (on validation set)")
    print("="*70)
    
    thresholds = np.arange(0.0, 1.05, 0.05)
    
    results = []
    
    for thresh in thresholds:
        # Count true positives and false positives
        tp = 0  # Predicted pothole, actually pothole
        fp = 0  # Predicted pothole, actually background
        fn = 0  # Missed pothole (predicted background, actually pothole)
        tn = 0  # Predicted background, actually background
        
        for item in all_data:
            pothole_conf = item['pothole_confidence']
            gt_label = item['gt_label']
            
            # Prediction: pothole if confidence >= threshold
            pred_label = 1 if pothole_conf >= thresh else 0
            
            if pred_label == 1 and gt_label == 1:
                tp += 1
            elif pred_label == 1 and gt_label == 0:
                fp += 1
            elif pred_label == 0 and gt_label == 1:
                fn += 1
            else:
                tn += 1
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'detections': tp + fp
        })
        
        print(f"\nThreshold: {thresh:.2f}")
        print(f"  Precision: {precision:.4f}  (TP={tp}, FP={fp})")
        print(f"  Recall:    {recall:.4f}  (TP={tp}, FN={fn})")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Detections: {tp + fp}")
    
    # Find best threshold by F1-score
    best_idx = np.argmax([r['f1'] for r in results])
    best_result = results[best_idx]
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"\nBest threshold (by F1-score): {best_result['threshold']:.2f}")
    print(f"  F1-score: {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    print(f"\nUse this in step1_apply_cnn.py:")
    print(f"  pothole_confidence_threshold = {best_result['threshold']:.2f}")
    
    return best_result['threshold'], results


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/checkpoints/best_model.pt"
    DATASET_PATH = "/dtu/datasets1/02516/potholes/"
    PROPOSALS_DIR = "scratch/proposals"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Get validation set predictions
    confidences, all_data = get_all_confidences(
        MODEL_PATH, PROPOSALS_DIR, DATASET_PATH, device, split='val'
    )
    
    # Analyze thresholds
    best_threshold, results = analyze_thresholds(all_data)
    
    print("\n" + "="*70)
    print("Summary statistics on validation set:")
    print("="*70)
    print(f"Total proposals: {len(all_data)}")
    print(f"Pothole proposals: {np.sum([1 for d in all_data if d['gt_label'] == 1])}")
    print(f"Background proposals: {np.sum([1 for d in all_data if d['gt_label'] == 0])}")
    print(f"\nConfidence score distribution:")
    print(f"  Min: {confidences.min():.4f}")
    print(f"  Max: {confidences.max():.4f}")
    print(f"  Mean: {confidences.mean():.4f}")
    print(f"  Median: {np.median(confidences):.4f}\n")