import json
import random
from pathlib import Path


def create_splits(dataset_path, train_ratio=0.6, test_ratio=0.2, seed=42):
    """
    Generates splits.json for the Potholes dataset.

    The remainder (1 - train_ratio - test_ratio) is assigned to validation.
    Asserts that there is at least one file in train, val and test.
    """
    base_path = Path(dataset_path)
    images_dir = base_path / "images"
    ann_dir = base_path / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")

    # Validate ratios
    if train_ratio < 0 or test_ratio < 0:
        raise ValueError("train_ratio and test_ratio must be non-negative")
    if train_ratio + test_ratio >= 1.0:
        raise ValueError("train_ratio + test_ratio must be < 1.0 (some remainder needed for validation)")

    # Collect image files that have matching XML annotation
    all_images = []
    print(f"Scanning {images_dir}...")
    for img_file in sorted(images_dir.glob("*.png")):  # adapt extension if needed
        xml_file = ann_dir / (img_file.stem + ".xml")
        if xml_file.exists():
            all_images.append(img_file.name)
        else:
            print(f"Warning: Skipping {img_file.name} (No XML found)")

    n_total = len(all_images)
    print(f"Found {n_total} valid image-xml pairs.")
    if n_total == 0:
        raise RuntimeError("No valid image-annotation pairs found.")

    # Shuffle
    random.seed(seed)
    random.shuffle(all_images)

    # Calculate counts
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    n_val = n_total - n_train - n_test  # remainder goes to validation

    # Explicit assertion to ensure each split has at least one item
    assert n_train > 0 and n_val > 0 and n_test > 0, (
        f"Insufficient items for splits with given ratios and dataset size (n_total={n_total}): "
        f"n_train={n_train}, n_val={n_val}, n_test={n_test}. "
        "Adjust ratios or use a larger dataset."
    )

    # Build splits
    train_files = all_images[:n_train]
    val_files = all_images[n_train : n_train + n_val]
    test_files = all_images[n_train + n_val : n_train + n_val + n_test]

    splits = {"train": train_files, "val": val_files, "test": test_files}

    # Save
    output_path = base_path / "splits.json"
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"✓ Saved splits.json to {output_path}")
    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")


if __name__ == "__main__":
    create_splits("/dtu/datasets1/02516/potholes/")