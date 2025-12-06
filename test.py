"""Step 1: Apply CNN to test proposals."""
import torch
import pickle
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from src.models.pothole_ResNet import PotholeClassifier
from src.training.helpers import load_checkpoint


def apply_cnn_to_test_proposals(model_path, proposals_dir, dataset_path, device):
    """
    Apply trained CNN to all test proposals and get predictions.
    
    Args:
        model_path: Path to trained model (best_model.pt)
        proposals_dir: Path to proposals directory
        dataset_path: Path to dataset (for loading images)
        device: Device to run on (cuda/cpu)
        
    Returns:
        dict: predictions for each image (ONLY pothole predictions, not all proposals)
    """
    
    print(f"\n{'='*70}")
    print("STEP 1: APPLY CNN TO TEST PROPOSALS")
    print(f"{'='*70}\n")
    
    # Load trained model
    print("Loading trained model...")
    model = PotholeClassifier(num_classes=2, pretrained=True).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded from: {model_path}")
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Best metrics: {checkpoint['metrics']}\n")
    
    # Load test proposals
    print("Loading test proposals...")
    proposals_file = Path(proposals_dir) / 'proposals_test.pkl'
    with open(proposals_file, 'rb') as f:
        proposals_dict = pickle.load(f)
    print(f"  ✓ Loaded {len(proposals_dict)} images with proposals\n")
    
    # Normalization stats (ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Configuration
    pothole_confidence_threshold = 0.15  # Lower threshold to catch more potholes
    
    # Run inference
    print("Running inference on test proposals...")
    predictions = {}
    
    total_proposals = 0
    total_potholes = 0
    
    with torch.no_grad():
        for image_filename, data in tqdm(proposals_dict.items(), desc="Inference"):
            proposals = data['proposals']
            
            if len(proposals) == 0:
                predictions[image_filename] = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': np.array([])
                }
                continue
            
            # Load image
            image_path = Path(dataset_path) / 'images' / image_filename
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract patches from proposals
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
                
                # Normalize
                patch = patch.astype(np.float32) / 255.0
                patch = (patch - mean) / std
                patch = torch.tensor(patch).permute(2, 0, 1).float()
                patches.append(patch)
            
            # Run inference
            if patches:
                patches_tensor = torch.stack(patches)

                batch_size = 16
                all_scores = []
                all_labels = []

                for i in range(0, len(patches_tensor), batch_size):
                    batch = patches_tensor[i:i+batch_size].to(device)
                    logits = model(batch)
                    probabilities = torch.softmax(logits, dim=1)
                    scores = probabilities[:, 1]  # Confidence for pothole class
                    labels = torch.argmax(logits, dim=1)
        
                    all_scores.append(scores.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                
                # Concatenate all batches before filtering
                scores_array = np.concatenate(all_scores, axis=0)
                labels_array = np.concatenate(all_labels, axis=0)
                
                # FIX: Keep pothole predictions based on CONFIDENCE SCORE (not just argmax)
                # This catches potholes even if the model is not 100% certain
                pothole_mask = scores_array >= pothole_confidence_threshold
                pothole_indices = np.where(pothole_mask)[0]
                
                if len(pothole_indices) > 0:
                    predictions[image_filename] = {
                        'boxes': np.array(proposals)[pothole_indices],
                        'scores': scores_array[pothole_indices],
                        'labels': labels_array[pothole_indices]
                    }
                    total_potholes += len(pothole_indices)
                else:
                    predictions[image_filename] = {
                        'boxes': np.array([]),
                        'scores': np.array([]),
                        'labels': np.array([])
                    }
                
                total_proposals += len(proposals)
    
    print(f"  ✓ Inference complete\n")
    
    # Print statistics
    print("Inference Statistics:")
    print(f"  Total proposals tested:        {total_proposals}")
    print(f"  Pothole predictions:           {total_potholes}")
    print(f"  Pothole confidence threshold:  {pothole_confidence_threshold}")
    if total_proposals > 0:
        print(f"  Detection rate:                {100*total_potholes/total_proposals:.1f}%\n")
    
    # Save predictions
    output_file = Path(proposals_dir) / 'test_predictions.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"  ✓ Predictions saved to: {output_file}\n")
    
    return predictions


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/checkpoints/best_model.pt"
    DATASET_PATH = "/dtu/datasets1/02516/potholes/"
    PROPOSALS_DIR = "scratch/proposals"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Apply CNN to test proposals
    predictions = apply_cnn_to_test_proposals(MODEL_PATH, PROPOSALS_DIR, DATASET_PATH, device)
    
    print("="*70)
    print("✓ Step 1 complete!")
    print("="*70)
    print("\nNext: Run NMS to remove overlapping boxes")