# utils/metrics.py

import numpy as np
from collections import defaultdict
import motmetrics as mm
import re


def iou(boxA, boxB):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(boxAArea + boxBArea - inter + 1e-6)

def compute_detection_metrics(gt_boxes, pred_boxes, iou_thresh=0.5):
    TP = FP = FN = 0
    frames_with_all = 0
    total_frames = len(gt_boxes)
    for f, gts in gt_boxes.items():
        preds = pred_boxes.get(f, [])
        matched = set()
        for pb in preds:
            match = False
            for i, gb in enumerate(gts):
                if i not in matched and iou(gb, pb) >= iou_thresh:
                    TP += 1
                    matched.add(i)
                    match = True
                    break
            if not match:
                FP += 1
        FN += len(gts) - len(matched)
        if len(matched) == len(gts):
            frames_with_all += 1

    p = TP/(TP+FP+1e-6)
    r = TP/(TP+FN+1e-6)
    f1 = 2*p*r/(p+r+1e-6)
    return {
        'precision': p,
        'recall': r,
        'f1': f1,
        'per_frame_detection_rate': frames_with_all/total_frames
    }

def compute_iou_stats(gt_boxes, pred_boxes):
    all_ious = []
    for f, gts in gt_boxes.items():
        for gb in gts:
            best = 0
            for pb in pred_boxes.get(f, []):
                best = max(best, iou(gb, pb))
            all_ious.append(best)
    return {'mean_iou': np.mean(all_ious), 'median_iou': np.median(all_ious)}

def compute_association_accuracy(gt_assoc, pred_assoc):
    total = correct = 0
    for key, gt_f in gt_assoc.items():
        if key in pred_assoc:
            total += 1
            if pred_assoc[key] == gt_f:
                correct += 1
    return correct/(total+1e-6)

def compute_tracking_metrics(gt_tracks, pred_tracks):
    """
    gt_tracks, pred_tracks: dict[frame] -> list of (track_id, box)
    Builds a distance matrix of (1 - IoU) for each frame.
    """
    mh = mm.metrics.create()
    acc = mm.MOTAccumulator(auto_id=False)
    for fname in sorted(gt_tracks):
        m = re.search(r'frame_(\d+)\.jpg', fname)
        if m:
            frameid = int(m.group(1)) - 1  # minus 1 if your video frames start at 0
        else:
            # fallback: use enumeration index
            frameid = None
        gt = gt_tracks[fname]
        pr = pred_tracks.get(fname, [])
        gt_ids   = [t[0] for t in gt]
        gt_boxes = [t[1] for t in gt]
        pr_ids   = [t[0] for t in pr]
        pr_boxes = [t[1] for t in pr]
        # compute cost matrix = 1 - IoU
        if gt_boxes and pr_boxes:
            D = np.zeros((len(gt_boxes), len(pr_boxes)), dtype=float)
            for i, gb in enumerate(gt_boxes):
                for j, pb in enumerate(pr_boxes):
                    D[i, j] = 1 - iou(gb, pb)
        else:
            D = np.zeros((len(gt_boxes), len(pr_boxes)), dtype=float)

        if frameid is not None:
            acc.update(
                np.array(gt_ids),
                np.array(pr_ids),
                D,
                frameid=frameid
            )
        else:
            acc.update(
                np.array(gt_ids),
                np.array(pr_ids),
                D
            )
    summary = mh.compute(
        acc,
        metrics=['mota','idf1','num_switches','mostly_tracked','mostly_lost'],
        name='overall'
    )
    return summary.to_dict()

def compute_stability_metrics(tracks_by_id):
    """
    tracks_by_id: dict[track_id] -> list of (frame, box)
    Computes drift, area variation, and angle smoothness per track.
    """
    results = {}
    for tid, recs in tracks_by_id.items():
        recs = sorted(recs, key=lambda x: x[0])
        centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for _, b in recs]
        areas   = [ (b[2]-b[0])*(b[3]-b[1]) for _, b in recs ]
        x0, y0  = centers[0]
        drifts  = [np.hypot(x-x0, y-y0) for x,y in centers]
        area_var = np.std(areas)/(np.mean(areas)+1e-6)
        vels     = [(centers[i+1][0]-centers[i][0],
                     centers[i+1][1]-centers[i][1])
                    for i in range(len(centers)-1)]
        angles   = [abs(np.arctan2(dy,dx)) for dx,dy in vels if dx or dy]
        ang_std  = np.std(angles) if angles else 0
        results[tid] = {
            'max_drift': max(drifts),
            'area_var_norm': area_var,
            'angle_std': ang_std
        }
    return results
