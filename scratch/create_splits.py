import json
import os
from pathlib import Path
import random

def create_splits_json(dataset_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    """
    Create splits.json file with train/validation/test split by scanning the images directory
    
    Args:
        dataset_path (str): Path to the potholes dataset
        train_ratio (float): Ratio of images for training (default 0.6 = 60%)
        val_ratio (float): Ratio of images for validation (default 0.2 = 20%)
        test_ratio (float): Ratio of images for testing (default 0.2 = 20%)
        random_seed (int): Random seed for reproducibility
    """
    
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Error: Ratios must sum to 1.0, but got {total_ratio}")
        return None
    
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images'
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg'}
    all_images = sorted([f.name for f in images_dir.iterdir() 
                        if f.suffix.lower() in image_extensions])
    
    if not all_images:
        print(f"❌ No images found in {images_dir}")
        return None
    
    print(f"Found {len(all_images)} images")
    
    # Split into train, validation, and test
    random.seed(random_seed)
    random.shuffle(all_images)
    
    train_end = int(len(all_images) * train_ratio)
    val_end = train_end + int(len(all_images) * val_ratio)
    
    train_files = all_images[:train_end]
    val_files = all_images[train_end:val_end]
    test_files = all_images[val_end:]
    
    # Create splits dictionary
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    # Save to JSON
    splits_file = dataset_path / 'splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ Created splits.json")
    print(f"  - Training images:   {len(train_files)} ({100*len(train_files)/len(all_images):.1f}%)")
    print(f"  - Validation images: {len(val_files)} ({100*len(val_files)/len(all_images):.1f}%)")
    print(f"  - Test images:       {len(test_files)} ({100*len(test_files)/len(all_images):.1f}%)")
    print(f"  - Total images:      {len(all_images)}")
    print(f"  - Saved to: {splits_file}")
    
    return splits


if __name__ == "__main__":
    DATASET_PATH = "/dtu/datasets1/02516/potholes/"
    SAVE_PATH = "/zhome/e2/6/224426/project/object-detection/scratch/"
    
    # Try to save in the dataset directory first
    # If that fails, save in current working directory or home directory
    original_create = create_splits_json
    
    def create_splits_json_safe(dataset_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
        result = original_create(dataset_path, train_ratio, val_ratio, test_ratio, random_seed)
        return result
    
    try:
        # Create the splits.json file with 60/20/20 split
        splits = create_splits_json_safe(DATASET_PATH, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    except PermissionError:
        print(f"\n⚠ No write permission to {DATASET_PATH}")
        print("Saving to current directory instead...")
        
        # Modify dataset_path for saving only
        import os
        dataset_path = Path(DATASET_PATH)
        images_dir = dataset_path / 'images'
        
        image_extensions = {'.png', '.jpg', '.jpeg'}
        all_images = sorted([f.name for f in images_dir.iterdir() 
                            if f.suffix.lower() in image_extensions])
        
        random.seed(42)
        random.shuffle(all_images)
        
        train_end = int(len(all_images) * 0.6)
        val_end = train_end + int(len(all_images) * 0.2)
        
        train_files = all_images[:train_end]
        val_files = all_images[train_end:val_end]
        test_files = all_images[val_end:]
        
        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        # Save to specified directory
        splits_file = Path(SAVE_PATH) / 'splits.json'
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"\n✓ Created splits.json in: {splits_file.absolute()}")
        print(f"  - Training images:   {len(train_files)} ({100*len(train_files)/len(all_images):.1f}%)")
        print(f"  - Validation images: {len(val_files)} ({100*len(val_files)/len(all_images):.1f}%)")
        print(f"  - Test images:       {len(test_files)} ({100*len(test_files)/len(all_images):.1f}%)")
        print(f"  - Total images:      {len(all_images)}")
    
    if splits:
        print("\n✓ Splits created successfully!")
        print(f"\nFirst 3 training images: {splits['train'][:3]}")
        print(f"First 3 validation images: {splits['val'][:3]}")
        print(f"First 3 test images: {splits['test'][:3]}")