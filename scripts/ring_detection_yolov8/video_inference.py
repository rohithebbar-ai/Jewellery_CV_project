"""
Run the fine-tuned YOLOv8 ring detector on a demo video
draw bounding box and confidence scores and display in real time.


import argparse
import cv2
from ultralytics import YOLO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mediapipe_utils import get_hand_landmarks
from utils.ring_finger_matcher import find_closest_landmark

def parse_args():
    p = argparse.ArgumentParser(description="Ring-on-hand video inference")
    p.add_argument(
        "--model",
        type=str,
        default='models/ring_detector/weights/best.pt',
        help="Path to your fine-tuned YOLOv8 .pt file"
    )
    p.add_argument(
        "--source",
        type=str,
        default="/home/zeus/workspace/Jewellery_CV_project/data/anna_demo_2.mp4",
        help="Path to your input data (video file) - .mp4,.avi, etc"
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.41,
        help="Detection confidence threshold"
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    p.add_argument(
        "--out",
        type=str,
        default="annotated_demo.mp4",
        help="Where to save the annotated output video"
    )

    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load the model
    model = YOLO(args.model)

    # 2) Open Video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open the video: {args.source}")
        return

    # ── CHANGED: set up VideoWriter with robust FOURCC lookup
    if hasattr(cv2, 'VideoWriter_fourcc'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif hasattr(cv2.VideoWriter, 'fourcc'):
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    else:
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    print(f"[INFO] Writing annotated video to: {args.out}")


    # 3) Process the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get hand landmarks
        hand_landmarks = get_hand_landmarks(frame)

        # Run detection on thr raw frame
        results = model(frame, conf=args.conf, iou=args.iou)[0]

        # Draw boxes + confidences
        for box in results.boxes:
            # box.xyxy is a tensor [[x1, y1, x2, y2]]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            ring_center = [(x1 + x2) / 2 / width, (y1 + y2) / 2 /height]

            #Find closest landmark (e.g ring finger)
            closest_id = find_closest_landmark(ring_center, hand_landmarks)
            finger_name = f"Landmark ID: {closest_id}" if closest_id is not None else "Unknown"
            

            # draw a rectangle
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 4)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Show
        out.write(frame)


    # cleanup
    cap.release()
    out.release()


if __name__ == "__main__":
    main()"""



"""
# 3D coordinates

import argparse, os, sys
import cv2
import mediapipe as mp
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mediapipe_utils import get_hand_landmarks
from utils.ring_finger_matcher import find_closest_landmark_3d
from utils.drawing_utils import draw_box_with_label

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ring_detector/weights/best.pt")
    p.add_argument("--source", default="data/anna_demo_2.mp4")
    p.add_argument("--conf",   type=float, default=0.41)
    p.add_argument("--iou",    type=float, default=0.45)
    p.add_argument("--out",    default="annotated_demo.mp4")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    cap   = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.source}")

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.out, fourcc, fps, (W, H))
    print(f"[INFO] Writing → {args.out}")

    mp_draw       = mp.solutions.drawing_utils
    mp_hands_mod  = mp.solutions.hands

    tracker = DeepSort(
        max_age=30,    # frames to keep “lost” tracks alive
        n_init=3,      # frames before confirming a new track
        max_cosine_distance=0.4,
        embedder="mobilenet",  # or your preferred small CNN
    )

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1) detect hands
        hand_landmarks = get_hand_landmarks(frame)

        # 2) draw them
        for hand in hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand, mp_hands_mod.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,128,255), thickness=1),
            )

        # 3) detect rings
        results = model(frame, conf=args.conf, iou=args.iou)[0]

        # 4) for each ring, find & draw its nearest finger landmark
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            # ring center in normalized coords
            ring_cx = ((x1 + x2) / 2) / W
            ring_cy = ((y1 + y2) / 2) / H

            hand_idx, lm_idx = find_closest_landmark_3d(
                                                (ring_cx, ring_cy),
                                                hand_landmarks,
                                                angle_thresh=0.6
                                            )

            if hand_idx is not None:
                lm = hand_landmarks[hand_idx].landmark[lm_idx]
                lm_px = (int(lm.x * W), int(lm.y * H))
                ring_c_px = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # draw connection line + landmark dot
                cv2.circle(frame, lm_px, 5, (0,0,255), -1)
                cv2.line(frame, ring_c_px, lm_px, (255,0,0), 2)

                label = f"{conf:.2f} | Finger {lm_idx}"
            else:
                label = f"{conf:.2f} | Unknown"

            draw_box_with_label(frame, (x1, y1, x2, y2), label)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
"""


#!/usr/bin/env python3
import argparse, os, sys
import cv2
import mediapipe as mp
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mediapipe_utils import get_hand_landmarks
from utils.ring_finger_matcher import find_closest_landmark_3d
from utils.drawing_utils import draw_box_with_label

# a small palette of distinct BGR colors for different tracks
PALETTE = [
    (255,  80,  80),
    ( 80, 255,  80),
    ( 80,  80, 255),
    (255, 255,  80),
    (255,  80, 255),
    ( 80, 255, 255),
    (128,   0,   0),
    (  0, 128,   0),
    (  0,   0, 128),
]

def finger_name(lm_idx):
    """Convert a MediaPipe landmark index into a human-readable finger name."""
    if lm_idx is None:
        return "Unknown"
    if lm_idx <= 4:
        return "Thumb"
    elif lm_idx <= 8:
        return "Index"
    elif lm_idx <= 12:
        return "Middle"
    elif lm_idx <= 16:
        return "Ring"
    else:
        return "Pinky"

def parse_args():
    p = argparse.ArgumentParser(description="Ring-on-hand tracking + finger assignment")
    p.add_argument("--model",  default="models/ring_detector/weights/best.pt",
                   help="Path to YOLOv8 ring detector (.pt)")
    p.add_argument("--source", default="data/anna_demo_2.mp4",
                   help="Path to input video")
    p.add_argument("--conf",   type=float, default=0.41,
                   help="YOLO detection confidence threshold")
    p.add_argument("--iou",    type=float, default=0.45,
                   help="YOLO NMS IoU threshold")
    p.add_argument("--out",    default="annotated_demo.mp4",
                   help="Where to save output video")
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    cap   = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    # prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter(args.out, fourcc, fps, (W, H))
    print(f"[INFO] Writing annotated video to: {args.out}")

    mp_draw      = mp.solutions.drawing_utils
    mp_hands_mod = mp.solutions.hands

    # initialize DeepSORT
    tracker = DeepSort(
        max_age=30,               # keep lost tracks for 30 frames
        n_init=3,                 # require 3 hits before confirmation
        max_cosine_distance=0.4,  # appearance distance threshold
        embedder="mobilenet",     # small CNN for appearance
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) detect hands & draw skeleton
        hand_landmarks = get_hand_landmarks(frame)
        for hand in hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand, mp_hands_mod.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,128,255), thickness=1),
            )

        # 2) run YOLO ring detection
        yolo_results = model(frame, conf=args.conf, iou=args.iou)[0]

        # 3) convert to DeepSORT’s expected (tlwh, confidence, class) format
        detections = []
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = x2 - x1
            h = y2 - y1
            conf_score = float(box.conf[0])
            detections.append(([x1, y1, w, h], conf_score, "ring"))

        # 4) update DeepSORT tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # 5) draw each confirmed track + finger assignment
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = int(track.track_id)  # ensure integer
            x1, y1, x2, y2 = map(int, track.to_tlbr())

            # compute ring center in normalized coords
            ring_cx = ((x1 + x2) / 2) / W
            ring_cy = ((y1 + y2) / 2) / H

            # 3D-aware finger matching
            hand_idx, lm_idx = find_closest_landmark_3d(
                (ring_cx, ring_cy),
                hand_landmarks,
                angle_thresh=0.6
            )

            # draw connection line + landmark dot if we found one
            if hand_idx is not None:
                lm = hand_landmarks[hand_idx].landmark[lm_idx]
                lm_px = (int(lm.x * W), int(lm.y * H))
                ring_center_px = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, lm_px, 6, (0, 0, 255), -1)
                cv2.line(frame, ring_center_px, lm_px, (255, 0, 0), 2)

            # assemble label: “ID3 | Ring” or “ID7 | Index”
            label = f"ID{tid} | {finger_name(lm_idx)}"

            # pick a stable color per track
            color = PALETTE[tid % len(PALETTE)]

            draw_box_with_label(
                frame,
                (x1, y1, x2, y2),
                label,
                color=color,
                box_thickness=4,
                font_scale=1.5,
                font_thickness=3
            )

        out.write(frame)

    cap.release()
    out.release()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
