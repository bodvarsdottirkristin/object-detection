import numpy as np
from IoU import iou 


def compute_ap(rec, prec):
    """
    VOC-style: integration over precision-recall curve.
    """
    rec = np.concatenate(([0.0], rec, [1.0]))
    prec = np.concatenate(([0.0], prec, [0.0]))

    # make precision monotonically decreasing
    for i in range(len(prec)-2, -1, -1):
        prec[i] = max(prec[i], prec[i+1])

    # find points where recall changes
    idx = np.where(rec[1:] != rec[:-1])[0]

    # sum over (Δrecall * precision)
    ap = np.sum((rec[idx+1] - rec[idx]) * prec[idx+1])
    return ap


def compute_map(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    pred_boxes: (img_id, class_id, conf, x1, y1, x2, y2)
    gt_boxes:   (img_id, class_id, x1, y1, x2, y2)
    """
    pred_boxes = np.array(pred_boxes)
    gt_boxes = np.array(gt_boxes)

    classes = np.unique(gt_boxes[:,1]).astype(int)
    ap_list = []

    for cls in classes:
        preds = pred_boxes[pred_boxes[:,1] == cls]
        gts = gt_boxes[gt_boxes[:,1] == cls]

        n_gt = len(gts)
        if n_gt == 0:
            continue

        # sort by confidence desc
        preds = preds[np.argsort(-preds[:,2])]

        # track TPs and FPs
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        matched = {}  # (img_id -> list of matched gt indices)

        for i, p in enumerate(preds):
            img_id, _, _, px1, py1, px2, py2 = p
            p_box = [px1, py1, px2, py2]

            gt_img = gts[gts[:,0] == img_id]
            if len(gt_img) == 0:
                fp[i] = 1
                continue

            ious = []
            for g in gt_img:
                gx1, gy1, gx2, gy2 = g[2:]
                ious.append(iou(p_box, [gx1, gy1, gx2, gy2]))
            ious = np.array(ious)

            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            gt_global_idx = np.where((gt_boxes == gts[max_iou_idx]).all(axis=1))[0][0]

            if max_iou >= iou_thresh:
                if (img_id, gt_global_idx) not in matched:
                    tp[i] = 1
                    matched[(img_id, gt_global_idx)] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        # precision-recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        rec = tp_cum / n_gt
        prec = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap = compute_ap(rec, prec)
        ap_list.append(ap)

    mAP = np.mean(ap_list) if ap_list else 0
    return mAP, ap_list