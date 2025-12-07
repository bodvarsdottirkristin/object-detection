import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


class PotholesDataset:
    """Class to load and explore the Potholes dataset"""

    def __init__(self, dataset_path, splits_path):
        """
        Initialize the dataset loader

        Args:
            dataset_path (str): Path to the potholes dataset directory
            splits_path (str): Path to the splits.json file
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.annotations_dir = self.dataset_path / "annotations"

        # Load splits from specified path
        with open(splits_path, "r") as f:
            self.splits = json.load(f)

        self.train_files = self.splits["train"]
        self.val_files = self.splits.get("val", [])
        self.test_files = self.splits["test"]

        print(f"✓ Dataset loaded from: {dataset_path}")
        print(f"✓ Splits loaded from: {splits_path}")
        print(f"  - Training images: {len(self.train_files)}")
        if self.val_files:
            print(f"  - Validation images: {len(self.val_files)}")
        print(f"  - Test images: {len(self.test_files)}")

    def parse_xml_annotation(self, xml_path):
        """
        Parse a single XML annotation file in PascalVOC format

        Args:
            xml_path (str): Path to the XML annotation file

        Returns:
            dict: Parsed annotation data including image info and bounding boxes
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image information
        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # Get bounding boxes
        bboxes = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            bbox = {
                "class": name,
                "xmin": int(float(bndbox.find("xmin").text)),
                "ymin": int(float(bndbox.find("ymin").text)),
                "xmax": int(float(bndbox.find("xmax").text)),
                "ymax": int(float(bndbox.find("ymax").text)),
            }
            bboxes.append(bbox)

        return {
            "filename": filename,
            "width": width,
            "height": height,
            "bboxes": bboxes,
        }

    def load_image_with_annotations(self, image_filename):
        """
        Load an image and its annotations

        Args:
            image_filename (str): Filename of the image

        Returns:
            tuple: (image, annotation_data) or (None, None) if not found
        """
        import cv2

        image_path = self.images_dir / image_filename
        # Try different extensions if needed
        if not image_path.exists():
            for ext in [".jpg", ".png", ".jpeg"]:
                alt_path = self.images_dir / (image_filename.split(".")[0] + ext)
                if alt_path.exists():
                    image_path = alt_path
                    break

        if not image_path.exists():
            print(f"⚠ Image not found: {image_path}")
            return None, None

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠ Could not read image: {image_path}")
            return None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations
        xml_filename = image_filename.rsplit(".", 1)[0] + ".xml"
        xml_path = self.annotations_dir / xml_filename

        if not xml_path.exists():
            print(f"⚠ Annotation not found: {xml_path}")
            return image, None

        annotation = self.parse_xml_annotation(str(xml_path))
        return image, annotation

    def visualize_annotations(self, image_filename, output_dir=None, figsize=(12, 8)):
        """
        Visualize an image with its ground-truth bounding boxes

        Args:
            image_filename (str): Filename of the image to visualize
            output_dir (str): Directory to save visualization. If None, displays with plt.show()
            figsize (tuple): Figure size for matplotlib
        """
        image, annotation = self.load_image_with_annotations(image_filename)

        if image is None:
            print(f"Could not load image: {image_filename}")
            return

        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)

        if annotation is not None and annotation["bboxes"]:
            for bbox in annotation["bboxes"]:
                # Draw rectangle
                x_min = bbox["xmin"]
                y_min = bbox["ymin"]
                width = bbox["xmax"] - bbox["xmin"]
                height = bbox["ymax"] - bbox["ymin"]

                rect = patches.Rectangle(
                    (x_min, y_min),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add label
                ax.text(
                    x_min,
                    y_min - 5,
                    bbox["class"],
                    color="red",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(facecolor="yellow", alpha=0.7),
                )

            title = f"{image_filename} ({len(annotation['bboxes'])} potholes)"
        else:
            title = f"{image_filename} (no annotations)"

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / f"viz_{image_filename}"
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            print(f"  ✓ Saved to: {output_path}")
            plt.close()
        else:
            plt.show()

    def get_dataset_statistics(self, split="train"):
        """
        Compute statistics about the dataset

        Args:
            split (str): 'train', 'val', or 'test'

        Returns:
            dict: Statistics about the dataset
        """
        if split == "train":
            files = self.train_files
        elif split == "val":
            files = self.val_files
        else:
            files = self.test_files

        stats = {
            "num_images": len(files),
            "num_potholes": 0,
            "avg_potholes_per_image": 0,
            "bbox_widths": [],
            "bbox_heights": [],
            "image_dimensions": [],
        }

        for filename in files:
            xml_filename = filename.rsplit(".", 1)[0] + ".xml"
            xml_path = self.annotations_dir / xml_filename

            if xml_path.exists():
                annotation = self.parse_xml_annotation(str(xml_path))
                num_bboxes = len(annotation["bboxes"])
                stats["num_potholes"] += num_bboxes
                stats["image_dimensions"].append(
                    (annotation["width"], annotation["height"])
                )

                for bbox in annotation["bboxes"]:
                    w = bbox["xmax"] - bbox["xmin"]
                    h = bbox["ymax"] - bbox["ymin"]
                    stats["bbox_widths"].append(w)
                    stats["bbox_heights"].append(h)

        if stats["num_images"] > 0:
            stats["avg_potholes_per_image"] = (
                stats["num_potholes"] / stats["num_images"]
            )

        return stats

    def print_statistics(self, split="train"):
        """Print formatted dataset statistics"""
        stats = self.get_dataset_statistics(split)

        print(f"\n{'=' * 60}")
        print(f"Dataset Statistics - {split.upper()} SET")
        print(f"{'=' * 60}")
        print(f"Number of images: {stats['num_images']}")
        print(f"Total number of potholes: {stats['num_potholes']}")
        print(f"Average potholes per image: {stats['avg_potholes_per_image']:.2f}")

        if stats["bbox_widths"]:
            print("\nBounding Box Widths:")
            print(f"  - Min: {min(stats['bbox_widths'])} px")
            print(f"  - Max: {max(stats['bbox_widths'])} px")
            print(f"  - Mean: {np.mean(stats['bbox_widths']):.2f} px")
            print(f"  - Std: {np.std(stats['bbox_widths']):.2f} px")

            print("\nBounding Box Heights:")
            print(f"  - Min: {min(stats['bbox_heights'])} px")
            print(f"  - Max: {max(stats['bbox_heights'])} px")
            print(f"  - Mean: {np.mean(stats['bbox_heights']):.2f} px")
            print(f"  - Std: {np.std(stats['bbox_heights']):.2f} px")

        if stats["image_dimensions"]:
            widths = [d[0] for d in stats["image_dimensions"]]
            heights = [d[1] for d in stats["image_dimensions"]]
            print("\nImage Dimensions:")
            print(f"  - Width: {min(widths)} - {max(widths)} px")
            print(f"  - Height: {min(heights)} - {max(heights)} px")

        print(f"{'=' * 60}\n")


# Example usage
if __name__ == "__main__":
    DATASET_PATH = "/dtu/datasets1/02516/potholes/"
    SPLITS_PATH = "/zhome/e2/6/224426/project/object-detection-1/scratch/splits.json"
    OUTPUT_DIR = "/zhome/e2/6/224426/project/object-detection-1/scratch/visualizations"

    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize the dataset
    try:
        dataset = PotholesDataset(DATASET_PATH, SPLITS_PATH)

        # Print statistics for all splits
        dataset.print_statistics("train")
        if dataset.val_files:
            dataset.print_statistics("val")
        dataset.print_statistics("test")

        # Visualize a few random examples from training set
        print("Visualizing random examples from training set...")
        import random

        random_files = random.sample(
            dataset.train_files, min(3, len(dataset.train_files))
        )

        for filename in random_files:
            print(f"Processing: {filename}")
            dataset.visualize_annotations(filename, output_dir=OUTPUT_DIR)

        print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nPlease make sure the paths are correct:")
        print(f"  Dataset: {DATASET_PATH}")
        print(f"  Splits: {SPLITS_PATH}")
