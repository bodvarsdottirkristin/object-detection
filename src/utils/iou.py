def iou(box1, box2):
    # Coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # If there is no overlap
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)

    # Areas of the two boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area

    return inter_area / union_area
