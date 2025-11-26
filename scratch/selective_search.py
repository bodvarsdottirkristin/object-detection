import os
import json
import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class SelectiveSeach:
    """Extract object proposals using Selective Search"""
    
    def __init__(self, dataset_path, splits_path, output_dir, max_img_size=800):
        """
        Initialize proposal extractor
        
        Args:
            dataset_path (str): Path to potholes dataset
            splits_path (str): Path to splits.json
            output_dir (str): Directory to save proposals
            max_img_size (int): Maximum image dimension for resizing
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / 'images'
        self.annotations_dir = self.dataset_path / 'annotations'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_img_size = max_img_size
        
        # Load splits
        with open(splits_path, 'r') as f:
            self.splits = json.load(f)
        
    
    def resize_image(self, image):
        """
        Resize image if it's larger than max_img_size
        
        Args:
            image: Input image
            
        Returns:
            tuple: (resized_image, scale_factor)
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > self.max_img_size:
            scale = self.max_img_size / max_dim
            new_h = int(h * scale)
            new_w = int(w * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized, scale
        else:
            return image, 1.0
    
    def extract_proposals_ss(self, image):
        """
        Extract proposals using Selective Search
        
        Args:
            image: Input image (BGR)
            
        Returns:
            list: List of bounding boxes [x1, y1, x2, y2]
        """
        # Create Selective Search object
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()  # Fast mode
        
        # Extract proposals
        proposals = ss.process()
        
        # Convert to list and filter small boxes
        proposals_list = []
        for prop in proposals:
            x, y, w, h = prop
            x2 = x + w
            y2 = y + h
            
            # Filter out very small proposals (less than 20x20)
            if w >= 20 and h >= 20:
                proposals_list.append([x, y, x2, y2])
        
        return proposals_list
    
    def process_image(self, image_filename):
        """
        Process single image and extract proposals
        
        Args:
            image_filename (str): Name of image file
            
        Returns:
            dict: Image info and proposals
        """
        image_path = self.images_dir / image_filename
        
        # Try different extensions
        if not image_path.exists():
            for ext in ['.jpg', '.png', '.jpeg']:
                alt_path = self.images_dir / (image_filename.split('.')[0] + ext)
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            return None
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Resize if needed
        resized_image, scale = self.resize_image(image)
        
        # Extract proposals using Selective Search
        proposals = self.extract_proposals_ss(resized_image)
        
        # Scale proposals back to original size
        if scale != 1.0:
            proposals = [[int(p[0]/scale), int(p[1]/scale), 
                         int(p[2]/scale), int(p[3]/scale)] for p in proposals]
        
        return {
            'image_filename': image_filename,
            'original_shape': (h, w),
            'resized_shape': resized_image.shape[:2],
            'scale': scale,
            'num_proposals': len(proposals),
            'proposals': proposals
        }
    
    def process_split(self, split='train'):
        """
        Process all images in a split
        
        Args:
            split (str): 'train', 'val', or 'test'
            
        Returns:
            dict: Statistics about extraction
        """
        files = self.splits[split]
        results = {}
        failed = []
        
        print(f"\nProcessing {split.upper()} set ({len(files)} images)...")
        
        for image_filename in tqdm(files, desc=f"{split} - Extracting proposals"):
            result = self.process_image(image_filename)
            
            if result is None:
                failed.append(image_filename)
            else:
                results[image_filename] = result
        
        # Save results
        output_file = self.output_dir / f'proposals_{split}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Print statistics
        total_proposals = sum(r['num_proposals'] for r in results.values())
        avg_proposals = total_proposals / len(results) if results else 0
        
        print(f"\n{split.upper()} SET RESULTS:")
        print(f"  - Images processed: {len(results)}")
        print(f"  - Images failed: {len(failed)}")
        print(f"  - Total proposals: {total_proposals}")
        print(f"  - Avg proposals per image: {avg_proposals:.1f}")
        print(f"  - Saved to: {output_file}")
        
        return {
            'split': split,
            'num_processed': len(results),
            'num_failed': len(failed),
            'total_proposals': total_proposals,
            'avg_proposals': avg_proposals,
            'failed_images': failed
        }
    
    def extract_all(self):
        """Extract proposals for all splits"""
        all_stats = {}
        
        for split in ['train', 'val', 'test']:
            if split in self.splits:
                stats = self.process_split(split)
                all_stats[split] = stats
        
        # Save summary
        summary_file = self.output_dir / 'extraction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        return all_stats


class PotholesProposal(Dataset):
    """PyTorch Dataset for loading proposals and their labels"""
    
    def __init__(self, dataset_path, annotations_dir, proposals_dict, iou_threshold=0.5):
        """
        Initialize proposal dataset
        
        Args:
            dataset_path (str): Path to dataset
            annotations_dir (str): Path to annotations
            proposals_dict (dict): Dictionary of proposals from pickle file
            iou_threshold (float): IoU threshold for positive/negative labels
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / 'images'
        self.annotations_dir = Path(annotations_dir)
        self.proposals_dict = proposals_dict
        self.iou_threshold = iou_threshold
        
        # Create list of (image_filename, proposal_idx) pairs
        self.samples = []
        for image_filename, data in proposals_dict.items():
            for prop_idx in range(len(data['proposals'])):
                self.samples.append((image_filename, prop_idx))
    
    def parse_xml_annotation(self, xml_path):
        """Parse XML annotation file"""
        import xml.etree.ElementTree as ET
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
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
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
    
    def get_label(self, proposal, ground_truth_boxes):
        """
        Assign label to proposal (1 = pothole, 0 = background)
        
        Args:
            proposal: [x1, y1, x2, y2]
            ground_truth_boxes: List of ground truth boxes
            
        Returns:
            int: 1 if pothole, 0 if background
        """
        if not ground_truth_boxes:
            return 0
        
        max_iou = max(self.compute_iou(proposal, gt) for gt in ground_truth_boxes)
        return 1 if max_iou >= self.iou_threshold else 0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get item: (proposal, label)
        
        Args:
            idx: Index
            
        Returns:
            tuple: (proposal_bbox, label)
        """
        image_filename, prop_idx = self.samples[idx]
        proposal = self.proposals_dict[image_filename]['proposals'][prop_idx]
        
        # Load ground truth
        xml_filename = image_filename.rsplit('.', 1)[0] + '.xml'
        xml_path = self.annotations_dir / xml_filename
        
        if xml_path.exists():
            gt_boxes = self.parse_xml_annotation(str(xml_path))
        else:
            gt_boxes = []
        
        label = self.get_label(proposal, gt_boxes)
        
        return {
            'proposal': torch.tensor(proposal, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'image_filename': image_filename
        }


# Run extraction
if __name__ == "__main__":
    DATASET_PATH = "/dtu/datasets1/02516/potholes/"
    SPLITS_PATH = "/zhome/e2/6/224426/project/object-detection-1/splits.json"
    OUTPUT_DIR = "/zhome/e2/6/224426/project/object-detection-1/scratch/proposals"
    MAX_IMG_SIZE = 800
    
    print("="*60)
    print("OBJECT PROPOSAL EXTRACTION")
    print("="*60)
    
    try:
        # Extract proposals
        extractor = SelectiveSeach(DATASET_PATH, SPLITS_PATH, OUTPUT_DIR, MAX_IMG_SIZE)
        print("\nUsing: Selective Search (Fast mode)")
        stats = extractor.extract_all()
        
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        
        # Example: Create dataloaders
        print("\n" + "="*60)
        print("CREATING DATALOADERS")
        print("="*60)
        
        # Load proposals
        with open(Path(OUTPUT_DIR) / 'proposals_train.pkl', 'rb') as f:
            train_proposals = pickle.load(f)
        
        # Create dataset and dataloader
        train_dataset = PotholesProposal(
            DATASET_PATH,
            Path(DATASET_PATH) / 'annotations',
            train_proposals,
            iou_threshold=0.5
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )
        
        print(f"\nTraining DataLoader created:")
        print(f"  - Total samples: {len(train_dataset)}")
        print(f"  - Batch size: 32")
        print(f"  - Batches per epoch: {len(train_loader)}")
        
        # Show example batch
        print("\nExample batch:")
        batch = next(iter(train_loader))
        print(f"  - Proposals shape: {batch['proposal'].shape}")
        print(f"  - Labels shape: {batch['label'].shape}")
        print(f"  - Label distribution: {torch.bincount(batch['label'])}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()