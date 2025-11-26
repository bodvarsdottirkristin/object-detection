import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

from src.datasets.parse_xml import parse_pothole_xml
from src.utils.iou import compute_iou


def main():
    base_path = Path("/dtu/datasets1/02516/potholes/")
    proposals_path = Path("src/datasets/proposals/proposals_train.pkl")
    output_db_path = Path("src/datasets/proposals/train_db.pkl")

    with open(proposals_path, "rb") as f:
        raw_proposals = pickle.load(f)

    labeled_data = []

    # Counters for balance
    pos_count = 0
    neg_count = 0

    for img_name, proposals in tqdm(raw_proposals.items(), desc="Labeling"):
        xml_path = base_path / "annotations" / (img_name.replace(".png", ".xml"))
        gt_boxes = parse_pothole_xml(xml_path)

        for prop_box in proposals:
            # Find best match
            max_iou = 0.0
            for gt in gt_boxes:
                iou = compute_iou(prop_box, gt)
                if iou > max_iou:
                    max_iou = iou

            # Label
            label = 1 if max_iou >= 0.5 else 0

            if label == 1:
                pos_count += 1
            else:
                neg_count += 1

            labeled_data.append(
                {
                    "img_path": str(base_path / "images" / img_name),
                    "box": prop_box,  # [x1, y1, x2, y2]
                    "label": label,
                }
            )

    print(f"Positives: {pos_count}, Negatives: {neg_count}")
    with open(output_db_path, "wb") as f:
        pickle.dump(labeled_data, f)


if __name__ == "__main__":
    main()
