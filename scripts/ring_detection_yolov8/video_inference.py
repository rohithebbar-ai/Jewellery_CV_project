#!/usr/bin/env python3
"""
Ring-on-hand tracking + segmentation inference script.
"""

import argparse
import os
import sys
import csv
import pickle
from time import perf_counter
from collections import defaultdict

import cv2
import mediapipe as mp
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# allow imports from utils folder
SCRIPT_DIR = os.path.dirname(__file__)
UTILS_DIR  = os.path.join(SCRIPT_DIR, 'utils')
sys.path.insert(0, UTILS_DIR)

from mediapipe_utils import get_hand_landmarks
from ring_finger_matcher import find_closest_landmark_3d
from drawing_utils import draw_box_with_label

# metrics
from metrics import (
    compute_detection_metrics,
    compute_iou_stats,
    compute_association_accuracy,
    compute_tracking_metrics,
    compute_stability_metrics,
)

PALETTE = [
    (255,  80,  80), ( 80, 255,  80), ( 80,  80, 255),
    (255, 255,  80), (255,  80, 255), ( 80, 255, 255),
    (128,   0,   0), (  0, 128,   0), (  0,   0, 128),
]

FINGER2ID = {"Thumb":0, "Index":1, "Middle":2, "Ring":3, "Pinky":4}

def finger_name(lm_idx):
    if lm_idx is None: return "Unknown"
    if lm_idx <= 4:    return "Thumb"
    if lm_idx <= 8:    return "Index"
    if lm_idx <= 12:   return "Middle"
    if lm_idx <= 16:   return "Ring"
    return "Pinky"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ring_detector/weights/best.pt")
    p.add_argument("--source", default="data/anna_demo_2.mp4")
    p.add_argument("--conf",   type=float, default=0.41)
    p.add_argument("--iou",    type=float, default=0.45)
    p.add_argument("--out",    default="annotated.mp4")
    p.add_argument("--csv",    default="tracks.csv")
    p.add_argument("--gt",     default="processed_gt.pkl")
    return p.parse_args()

def main():
    args = parse_args()

    # load models
    model = YOLO(args.model)
    cap   = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.source}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(args.out, fourcc, fps, (W, H))
    print(f"[INFO] Writing → {args.out}")

    csv_file = open(args.csv, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(["frame_idx","track_id","x1","y1","x2","y2","conf","finger"])

    mp_draw      = mp.solutions.drawing_utils
    mp_hands_mod = mp.solutions.hands
    tracker = DeepSort(max_age=30, n_init=3,
                       max_cosine_distance=0.4,
                       embedder="mobilenet")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = perf_counter()

        # hand landmarks
        hands = get_hand_landmarks(frame)
        for hand in hands:
            mp_draw.draw_landmarks(
                frame, hand, mp_hands_mod.HAND_CONNECTIONS,
                mp_draw.DrawingSpec((0,255,255),1,2),
                mp_draw.DrawingSpec((0,128,255),1)
            )

        # YOLO detection
        yres = model(frame, conf=args.conf, iou=args.iou)[0]
        dets = []
        for box in yres.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            w,h = x2-x1, y2-y1
            conf_score = float(box.conf[0])
            dets.append(([x1,y1,w,h], conf_score, "ring"))

        # DeepSORT
        tracks = tracker.update_tracks(dets, frame=frame)

        # draw & log
        for t in tracks:
            if not t.is_confirmed(): continue
            tid       = int(t.track_id)
            x1,y1,x2,y2 = map(int, t.to_tlbr())
            color     = PALETTE[tid % len(PALETTE)]
            cx,cy     = (x1+x2)//2, (y1+y2)//2
            rcx,rcy   = cx/W, cy/H

            # finger match
            hidx,lmidx = find_closest_landmark_3d((rcx,rcy), hands, angle_thresh=0.6)
            if hidx is not None:
                lm = hands[hidx].landmark[lmidx]
                lpx = (int(lm.x*W), int(lm.y*H))
                cv2.circle(frame, lpx, 5, (0,0,255), -1)
                cv2.line(frame, (cx,cy), lpx, (255,0,0),2)

            finger = finger_name(lmidx)
            draw_box_with_label(
                frame, (x1,y1,x2,y2),
                f"ID{tid} | {finger}",
                color=color, box_thickness=4,
                font_scale=1.0, font_thickness=2
            )
            conf_val = getattr(t, "detection_confidence", 0.0)
            writer.writerow([frame_idx, tid, x1,y1,x2,y2, f"{conf_val:.2f}", finger])

        out_vid.write(frame)
        dt = (perf_counter()-t0)*1000
        if dt>66: print(f"[WARN] Frame {frame_idx} took {dt:.1f}ms")
        frame_idx += 1

    # finish
    cap.release()
    out_vid.release()
    csv_file.close()
    print("[INFO] Inference done. Computing metrics…")

    # load GT
    with open(args.gt,'rb') as f:
        gt = pickle.load(f)
    gt_boxes     = gt['gt_boxes']
    gt_assoc     = gt['gt_assoc']
    gt_tracks    = gt['gt_tracks']
    tracks_by_id = gt['tracks_by_id']

    def idx_to_fname(idx):
        return f"frame_{idx+1:06d}.jpg"
    # map frame_idx→filename
    #frame_fnames = sorted(gt_boxes.keys())

    # parse predictions
    pred_boxes  = defaultdict(list)
    pred_assoc  = {}
    pred_tracks = defaultdict(list)
    pred_by_id  = defaultdict(list)

    with open(args.csv) as f:
        rd = csv.DictReader(f)
        for row in rd:
            idx = int(row['frame_idx'])
            tid = int(row['track_id'])
            x1,y1,x2,y2 = map(int,(row['x1'],row['y1'],row['x2'],row['y2']))
            finger = row['finger']

            fname = idx_to_fname(idx)
            if fname not in gt_boxes:
                continue
            pred_boxes[fname].append([x1,y1,x2,y2])
            pred_assoc[(fname,tid)] = FINGER2ID.get(finger,-1)
            pred_tracks[fname].append((tid,[x1,y1,x2,y2]))
            pred_by_id[tid].append((fname,[x1,y1,x2,y2]))
    
    # ─── DEBUG: Check predictions on each GT frame ───────────────────────
    print(f"[DEBUG] GT has {len(gt_boxes)} annotated frames")
    for f in gt_boxes:
        print(f"[DEBUG] Frame '{f}' → #predictions: {len(pred_boxes.get(f, []))}")

    
    print("Sample pred_boxes:", list(pred_boxes.items())[:5])
    print("Sample gt_boxes:   ", list(gt_boxes.items())[:5])

    # compute metrics
    detm   = compute_detection_metrics(gt_boxes, pred_boxes, iou_thresh=0.5)
    ioum   = compute_iou_stats(gt_boxes, pred_boxes)
    assocm = compute_association_accuracy(gt_assoc, pred_assoc)
    motm   = compute_tracking_metrics(gt_tracks, pred_tracks)
    stabm  = compute_stability_metrics(tracks_by_id)

    print("\n=== METRICS SUMMARY ===")
    print("Detection      :", detm)
    print("IoU stats      :", ioum)
    print("Assoc Accuracy :", assocm)
    print("Tracking       :", motm)
    print("Stability      :", stabm)
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
