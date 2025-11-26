import cv2
import pickle
import json
from tqdm import tqdm
from pathlib import Path


def get_proposals(img_path, max_size=800):
    """Runs Selective Search on a single image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return [], (0, 0)

    # 1. Resize for speed
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # 2. Run Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()  # Returns [x, y, w, h]

    # 3. Scale back to original size
    proposals = []
    for x, y, w, h in rects:
        # Filter small noise
        if w < 20 or h < 20:
            continue

        real_box = [
            int(x / scale),
            int(y / scale),
            int((x + w) / scale),
            int((y + h) / scale),
        ]  # Convert to [x1, y1, x2, y2]
        proposals.append(real_box)

    return proposals, (h, w)


def main():
    # Update these paths to match your environment
    base_path = Path("/dtu/datasets1/02516/potholes/")
    output_dir = Path("src/datasets/proposals/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Splits
    with open("splits.json") as f:
        splits = json.load(f)

    # Iterate over Train, Val, and Test
    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            raise ValueError(f"Split {split_name} not found in splits.json")

        print(f"Processing {split_name} split ({len(splits[split_name])} images)...")
        split_proposals = {}

        for img_name in tqdm(splits[split_name], desc=f"SS {split_name}"):
            img_path = base_path / "images" / img_name
            boxes, _ = get_proposals(img_path)
            split_proposals[img_name] = boxes

        # Save separate files for cleanliness: proposals_train.pkl, proposals_val.pkl, etc.
        save_path = output_dir / f"proposals_{split_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(split_proposals, f)

        print(f"✓ Saved {split_name} proposals to {save_path}")


if __name__ == "__main__":
    main()
