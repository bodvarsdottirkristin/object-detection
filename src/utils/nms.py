import numpy as np
from IoU import iou


def nms(boxes, scores, iou_thresh=0.5, score_thresh=0.0):
    """
    boxes: np.array of shape [N, 4]  (x1, y1, x2, y2)
    scores: np.array of shape [N]    (confidence for each box)
    iou_thresh: IoU threshold for suppression
    score_thresh: discard boxes with very low score
    """

    # Filter out boxes with low confidence
    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]

    # If nothing is left after filtering
    if len(scores) == 0:
        return []

    # Get indices sorted by descending score
    indices = np.argsort(scores)[::-1]

    selected_indices = []

    while len(indices) > 0:
        # Take the index of the box with the highest score
        current = indices[0]
        selected_indices.append(current)

        remaining = []
        # Compare this box with the remaining boxes
        for idx in indices[1:]:
            iou_val = iou(boxes[current], boxes[idx])
            # Keep boxes that do not overlap too much
            if iou_val <= iou_thresh:
                remaining.append(idx)

        indices = np.array(remaining, dtype=int)

    # Returns indices with respect to the filtered arrays
    # If you need indices with respect to the original arrays,
    # you should keep a mapping from original to filtered indices.
    return selected_indices


# ===== Example usage =====
if __name__ == "__main__":
    # 4 boxes [x1, y1, x2, y2]
    boxes = np.array(
        [
            [10, 10, 50, 50],  # Box A
            [12, 12, 48, 48],  # Box B (very close to A)
            [100, 100, 150, 150],  # Box C (separate)
            [105, 105, 148, 148],  # Box D (overlaps with C)
        ]
    )

    # Confidence scores for each box
    scores = np.array([0.9, 0.85, 0.8, 0.6])

    selected = nms(boxes, scores, iou_thresh=0.5, score_thresh=0.0)

    print("Selected indices:", selected)
    print("Selected boxes:\n", boxes[selected])
    print("Selected scores:", scores[selected])
