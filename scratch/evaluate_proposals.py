import os
import json
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

class ProposalEvaluator:
    """Evaluate object proposals quality"""
    
    def __init__(self, dataset_path, proposals_dir, output_dir):
        """
        Initialize evaluator
        
        Args:
            dataset_path (str): Path to potholes dataset
            proposals_dir (str): Directory with proposal pickle files
            output_dir (str): Directory to save evaluation results
        """
        self.dataset_path = Path(dataset_path)
        self.annotations_dir = self.dataset_path / 'annotations'
        self.proposals_dir = Path(proposals_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Initialized ProposalEvaluator")
        print(f"  - Dataset: {dataset_path}")
        print(f"  - Proposals: {proposals_dir}")
        print(f"  - Output: {output_dir}")
    
    def parse_xml_annotation(self, xml_path):
        """Parse XML annotation file"""
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
        """Compute IoU between two boxes"""
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
    
    def evaluate_image(self, image_filename, proposals, iou_threshold=0.5):
        """
        Evaluate proposals for single image
        
        Args:
            image_filename (str): Image filename
            proposals (list): List of proposals [x1, y1, x2, y2]
            iou_threshold (float): IoU threshold for positive label
            
        Returns:
            dict: Evaluation metrics for this image
        """
        # Load ground truth
        xml_filename = image_filename.rsplit('.', 1)[0] + '.xml'
        xml_path = self.annotations_dir / xml_filename
        
        if not xml_path.exists():
            return None
        
        gt_boxes = self.parse_xml_annotation(str(xml_path))
        
        if not gt_boxes:
            return None
        
        # For each ground truth box, find best matching proposal
        matched_proposals = []
        ious_per_gt = []
        
        for gt_box in gt_boxes:
            best_iou = 0.0
            best_prop_idx = -1
            
            for prop_idx, proposal in enumerate(proposals):
                iou = self.compute_iou(gt_box, proposal)
                if iou > best_iou:
                    best_iou = iou
                    best_prop_idx = prop_idx
            
            ious_per_gt.append(best_iou)
            if best_iou >= iou_threshold:
                matched_proposals.append(best_prop_idx)
        
        # Count positive proposals (overlap with GT)
        num_positive = 0
        for proposal in proposals:
            max_iou = max([self.compute_iou(proposal, gt) for gt in gt_boxes])
            if max_iou >= iou_threshold:
                num_positive += 1
        
        # Recall: how many GT boxes are covered by proposals?
        recall = len(matched_proposals) / len(gt_boxes) if gt_boxes else 0.0
        
        return {
            'image_filename': image_filename,
            'num_gt_boxes': len(gt_boxes),
            'num_proposals': len(proposals),
            'num_positive_proposals': num_positive,
            'num_matched_gt': len(matched_proposals),
            'recall': recall,
            'avg_iou_with_gt': np.mean(ious_per_gt) if ious_per_gt else 0.0,
            'max_iou_with_gt': np.max(ious_per_gt) if ious_per_gt else 0.0
        }
    
    def evaluate_split(self, split='train', iou_threshold=0.5):
        """
        Evaluate all proposals in a split
        
        Args:
            split (str): 'train', 'val', or 'test'
            iou_threshold (float): IoU threshold
            
        Returns:
            dict: Evaluation results and statistics
        """
        proposals_file = self.proposals_dir / f'proposals_{split}.pkl'
        
        if not proposals_file.exists():
            print(f"⚠ Proposals file not found: {proposals_file}")
            return None
        
        # Load proposals
        with open(proposals_file, 'rb') as f:
            proposals_dict = pickle.load(f)
        
        print(f"\nEvaluating {split.upper()} set...")
        
        results = []
        for image_filename, data in tqdm(proposals_dict.items(), desc=f"{split} - Evaluating"):
            eval_result = self.evaluate_image(
                image_filename,
                data['proposals'],
                iou_threshold
            )
            if eval_result:
                results.append(eval_result)
        
        if not results:
            return None
        
        # Aggregate statistics
        stats = {
            'split': split,
            'num_images': len(results),
            'total_gt_boxes': sum(r['num_gt_boxes'] for r in results),
            'total_proposals': sum(r['num_proposals'] for r in results),
            'total_positive_proposals': sum(r['num_positive_proposals'] for r in results),
            'total_matched_gt': sum(r['num_matched_gt'] for r in results),
            'avg_proposals_per_image': np.mean([r['num_proposals'] for r in results]),
            'avg_positive_per_image': np.mean([r['num_positive_proposals'] for r in results]),
            'recall': np.mean([r['recall'] for r in results]),
            'avg_iou': np.mean([r['avg_iou_with_gt'] for r in results]),
            'details': results
        }
        
        return stats
    
    def evaluate_all(self, iou_threshold=0.5):
        """Evaluate all splits"""
        all_results = {}
        
        for split in ['train', 'val', 'test']:
            stats = self.evaluate_split(split, iou_threshold)
            if stats:
                all_results[split] = stats
        
        return all_results
    
    def print_statistics(self, stats):
        """Print evaluation statistics"""
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS - {stats['split'].upper()} SET")
        print(f"{'='*70}")
        print(f"Number of images: {stats['num_images']}")
        print(f"Total ground truth boxes: {stats['total_gt_boxes']}")
        print(f"Total proposals extracted: {stats['total_proposals']}")
        print(f"Total positive proposals (IoU > 0.5): {stats['total_positive_proposals']}")
        print(f"Total matched ground truth boxes: {stats['total_matched_gt']}")
        
        print(f"\nAverage per image:")
        print(f"  - Proposals: {stats['avg_proposals_per_image']:.1f}")
        print(f"  - Positive proposals: {stats['avg_positive_per_image']:.1f}")
        print(f"  - Recall: {stats['recall']:.1%}")
        print(f"  - Avg IoU with GT: {stats['avg_iou']:.3f}")
        
        print(f"\nProposal efficiency:")
        ratio = stats['total_positive_proposals'] / stats['total_proposals'] if stats['total_proposals'] > 0 else 0
        print(f"  - Positive / Total ratio: {ratio:.3f} ({ratio*100:.1f}%)")
        print(f"{'='*70}\n")
    
    def plot_results(self, all_results):
        """Plot evaluation results"""
        splits = list(all_results.keys())
        recalls = [all_results[s]['recall'] for s in splits]
        avg_ious = [all_results[s]['avg_iou'] for s in splits]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Recall plot
        axes[0].bar(splits, recalls, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_ylabel('Recall', fontsize=12)
        axes[0].set_title('Proposal Recall by Split', fontsize=12, fontweight='bold')
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(recalls):
            axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
        
        # IoU plot
        axes[1].bar(splits, avg_ious, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_ylabel('Average IoU', fontsize=12)
        axes[1].set_title('Proposal Quality (Avg IoU) by Split', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(avg_ious):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / 'evaluation_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_file}")
        plt.close()
    
    def analyze_proposal_counts(self, all_results):
        """Analyze how many proposals are needed"""
        print(f"\n{'='*70}")
        print("PROPOSAL COUNT ANALYSIS")
        print(f"{'='*70}")
        
        for split, stats in all_results.items():
            avg_props = stats['avg_proposals_per_image']
            avg_positive = stats['avg_positive_per_image']
            
            print(f"\n{split.upper()} SET:")
            print(f"  - Average proposals per image: {avg_props:.0f}")
            print(f"  - Average positive proposals: {avg_positive:.0f}")
            print(f"  - Ratio (positive/total): {avg_positive/avg_props:.1%}")
            print(f"  - Recommendation: Keep top {int(avg_props * 0.5):.0f} proposals (50%)")
            print(f"                   or top {int(avg_props * 0.3):.0f} proposals (30%)")
        
        print(f"\n{'='*70}\n")


# Run evaluation
if __name__ == "__main__":
    DATASET_PATH = "/dtu/datasets1/02516/potholes/"
    PROPOSALS_DIR = "/zhome/e2/6/224426/project/object-detection-1/scratch/proposals"
    OUTPUT_DIR = "/zhome/e2/6/224426/project/object-detection-1/scratch/evaluation"
    
    print("="*70)
    print("PROPOSAL EVALUATION - PROJECT 4.1.3")
    print("="*70)
    
    try:
        evaluator = ProposalEvaluator(DATASET_PATH, PROPOSALS_DIR, OUTPUT_DIR)
        
        # Evaluate all splits
        all_results = evaluator.evaluate_all(iou_threshold=0.5)
        
        # Print results
        for split, stats in all_results.items():
            evaluator.print_statistics(stats)
        
        # Save results to JSON
        results_file = Path(OUTPUT_DIR) / 'evaluation_results.json'
        results_to_save = {}
        for split, stats in all_results.items():
            results_to_save[split] = {
                k: v for k, v in stats.items() if k != 'details'
            }
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"✓ Results saved to: {results_file}")
        
        # Plot results
        evaluator.plot_results(all_results)
        
        # Analyze proposal counts
        evaluator.analyze_proposal_counts(all_results)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()