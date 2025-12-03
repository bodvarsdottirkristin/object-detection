"""Generate labels for proposals based on ground-truth annotations."""
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


def parse_xml_annotation(xml_path):
    """Parse XML annotation file and extract bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        bbox = [
            int(float(bndbox.find('xmin').text)),
            int(float(bndbox.find('ymin').text)),
            int(float(bndbox.find('xmax').text)),
            int(float(bndbox.find('ymax').text))
        ]
        bboxes.append(bbox)
    
    return bboxes


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def assign_label(proposal, ground_truth_boxes, iou_threshold=0.5):
    """
    Assign label to proposal (1 = pothole, 0 = background)
    
    Args:
        proposal: [x1, y1, x2, y2]
        ground_truth_boxes: List of ground truth boxes
        iou_threshold: Threshold for positive label
        
    Returns:
        int: 1 if pothole, 0 if background
    """
    if not ground_truth_boxes:
        return 0
    
    max_iou = max(compute_iou(proposal, gt) for gt in ground_truth_boxes)
    return 1 if max_iou >= iou_threshold else 0


def generate_labels(dataset_path, proposals_dir, output_dir, iou_threshold=0.5):
    """Generate labels for all proposals."""
    
    dataset_path = Path(dataset_path)
    annotations_dir = dataset_path / 'annotations'
    proposals_dir = Path(proposals_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING LABELS FOR PROPOSALS")
    print(f"{'='*60}\n")
    
    for split in ['train', 'val', 'test']:
        print(f"Processing {split.upper()} set...")
        
        # Load proposals
        proposals_file = proposals_dir / f'proposals_{split}.pkl'
        with open(proposals_file, 'rb') as f:
            proposals_dict = pickle.load(f)
        
        # Generate labels
        labels_dict = {}
        
        for image_filename in tqdm(proposals_dict.keys(), desc=f"{split} - Generating labels"):
            proposals = proposals_dict[image_filename]['proposals']
            
            # Load ground truth
            xml_filename = image_filename.rsplit('.', 1)[0] + '.xml'
            xml_path = annotations_dir / xml_filename
            
            if xml_path.exists():
                gt_boxes = parse_xml_annotation(str(xml_path))
            else:
                gt_boxes = []
            
            # Assign labels to all proposals
            labels = [assign_label(p, gt_boxes, iou_threshold) for p in proposals]
            labels_dict[image_filename] = labels
        
        # Save labels
        labels_file = output_dir / f'labels_{split}.pkl'
        with open(labels_file, 'wb') as f:
            pickle.dump(labels_dict, f)
        
        # Print statistics
        total_labels = sum(len(labels) for labels in labels_dict.values())
        total_potholes = sum(sum(labels) for labels in labels_dict.values())
        total_background = total_labels - total_potholes
        
        print(f"  ✓ Saved to: {labels_file}")
        print(f"    - Total proposals: {total_labels}")
        print(f"    - Pothole proposals: {total_potholes}")
        print(f"    - Background proposals: {total_background}")
        print(f"    - Class imbalance ratio: {total_background/total_potholes if total_potholes > 0 else 0:.1f}x\n")
    
    print(f"{'='*60}")
    print("✓ Labels generation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    DATASET_PATH = "/dtu/datasets1/02516/potholes/"
    PROPOSALS_DIR = "scratch/proposals"
    OUTPUT_DIR = "scratch/proposals"
    
    generate_labels(DATASET_PATH, PROPOSALS_DIR, OUTPUT_DIR, iou_threshold=0.5)