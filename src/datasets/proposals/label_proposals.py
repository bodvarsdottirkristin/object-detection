import pickle
from pathlib import Path
from tqdm import tqdm

from src.datasets.parse_xml import parse_pothole_xml
from src.utils.iou import compute_iou


def main():
    base_path = Path("/dtu/datasets1/02516/potholes/")
    proposals_path = Path("src/datasets/proposals/proposals_train.pkl")
    output_db_path = Path("src/datasets/proposals/train_db.pkl")

    # Detect split type from filename
    split_type = (
        "train"
        if "train" in proposals_path.name
        else ("val" if "val" in proposals_path.name else "test")
    )

    with open(proposals_path, "rb") as f:
        raw_proposals = pickle.load(f)

    labeled_data = []
    pos_count = 0
    neg_count = 0

    for img_name, proposals in tqdm(raw_proposals.items(), desc="Labeling"):
        # Limit to top 1000 proposals only for train split
        if split_type == "train":
            proposals = proposals[:1000]

        xml_path = base_path / "annotations" / (img_name.replace(".png", ".xml"))
        gt_boxes = parse_pothole_xml(xml_path)

        for prop_box in proposals:
            max_iou = 0.0
            for gt in gt_boxes:
                iou = compute_iou(prop_box, gt)
                if iou > max_iou:
                    max_iou = iou

            label = 1 if max_iou >= 0.5 else 0

            if label == 1:
                pos_count += 1
            else:
                neg_count += 1

            labeled_data.append(
                {
                    "img_path": str(base_path / "images" / img_name),
                    "box": prop_box,
                    "label": label,
                }
            )

    print(f"Positives: {pos_count}, Negatives: {neg_count}")
    with open(output_db_path, "wb") as f:
        pickle.dump(labeled_data, f)

if __name__ == "__main__":
    main()