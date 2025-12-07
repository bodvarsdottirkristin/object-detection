import os
import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
import numpy as np

DATASET_PATH = "/dtu/datasets1/02516/potholes/"
SPLITS_PATH = "splits.json"
LABELED_PROPOSALS_PATH = "src/datasets/proposals/train_db.pkl"

def get_image_stats(images_dir):
    sizes = []
    formats = set()
    for fname in os.listdir(images_dir):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            try:
                import cv2
                img = cv2.imread(os.path.join(images_dir, fname))
                if img is not None:
                    h, w = img.shape[:2]
                    sizes.append((w, h))
                    formats.add(fname.split('.')[-1])
            except Exception as e:
                print(f"Error reading {fname}: {e}")
    sizes_np = np.array(sizes)
    stats = {
        "num_images": len(sizes),
        "formats": list(formats),
        "min_width": int(sizes_np[:,0].min()) if sizes else None,
        "max_width": int(sizes_np[:,0].max()) if sizes else None,
        "min_height": int(sizes_np[:,1].min()) if sizes else None,
        "max_height": int(sizes_np[:,1].max()) if sizes else None,
        "mean_width": float(sizes_np[:,0].mean()) if sizes else None,
        "mean_height": float(sizes_np[:,1].mean()) if sizes else None,
    }
    return stats

def get_annotation_stats(ann_dir, splits):
    box_counts = []
    class_counts = Counter()
    box_sizes = []
    missing_xml = []
    for split, files in splits.items():
        for fname in files:
            xml_path = os.path.join(ann_dir, fname.replace(".png", ".xml"))
            if not os.path.exists(xml_path):
                missing_xml.append(fname)
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()
            boxes = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                class_counts[name] += 1
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])
                box_sizes.append((xmax-xmin, ymax-ymin))
            box_counts.append(len(boxes))
    box_sizes_np = np.array(box_sizes)
    stats = {
        "total_boxes": sum(box_counts),
        "boxes_per_image": {
            "min": int(np.min(box_counts)) if box_counts else None,
            "max": int(np.max(box_counts)) if box_counts else None,
            "mean": float(np.mean(box_counts)) if box_counts else None,
        },
        "box_width": {
            "min": int(box_sizes_np[:,0].min()) if box_sizes else None,
            "max": int(box_sizes_np[:,0].max()) if box_sizes else None,
            "mean": float(box_sizes_np[:,0].mean()) if box_sizes else None,
        },
        "box_height": {
            "min": int(box_sizes_np[:,1].min()) if box_sizes else None,
            "max": int(box_sizes_np[:,1].max()) if box_sizes else None,
            "mean": float(box_sizes_np[:,1].mean()) if box_sizes else None,
        },
        "class_counts": dict(class_counts),
        "missing_xml": missing_xml,
    }
    return stats

def get_split_stats(splits_path):
    with open(splits_path) as f:
        splits = json.load(f)
    stats = {split: len(files) for split, files in splits.items()}
    return splits, stats

def get_proposal_label_stats(labeled_db_path):
    with open(labeled_db_path, "rb") as f:
        samples = pickle.load(f)
    label_counts = Counter()
    for s in samples:
        label_counts[s["label"]] += 1
    stats = {
        "total_proposals": len(samples),
        "positive": label_counts.get(1, 0),
        "negative": label_counts.get(0, 0),
        "positive_ratio": label_counts.get(1, 0) / len(samples) if samples else None,
    }
    return stats

def main():
    print("==== DATASET SUMMARY ====")
    images_dir = os.path.join(DATASET_PATH, "images")
    ann_dir = os.path.join(DATASET_PATH, "annotations")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Images dir:   {images_dir}")
    print(f"Annotations:  {ann_dir}")

    # Image stats
    img_stats = get_image_stats(images_dir)
    print("\nImage stats:")
    for k, v in img_stats.items():
        print(f"  {k}: {v}")

    # Splits
    splits, split_stats = get_split_stats(SPLITS_PATH)
    print("\nSplit sizes:")
    for k, v in split_stats.items():
        print(f"  {k}: {v}")

    # Annotation stats
    ann_stats = get_annotation_stats(ann_dir, splits)
    print("\nAnnotation stats:")
    for k, v in ann_stats.items():
        print(f"  {k}: {v}")

    # Proposal label stats
    if os.path.exists(LABELED_PROPOSALS_PATH):
        prop_stats = get_proposal_label_stats(LABELED_PROPOSALS_PATH)
        print("\nLabeled proposal stats (train):")
        for k, v in prop_stats.items():
            print(f"  {k}: {v}")
    else:
        print("\nLabeled proposal stats: train_db.pkl not found.")

    print("\nPotential issues:")
    if ann_stats["missing_xml"]:
        print(f"  Missing XML for {len(ann_stats['missing_xml'])} images.")
    else:
        print("  No missing annotation files detected.")

if __name__ == "__main__":
    main()