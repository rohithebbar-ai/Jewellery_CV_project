#!/usr/bin/env python3
# scripts/hough_video_prototype.py

import os
import json
import cv2
import mediapipe as mp
import numpy as np

# 1) Load config
"""BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, 'config.json')) as f:
    cfg = json.load(f)"""
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, 'config_mediapipe.json')
with open(CONFIG_PATH, 'r') as f:
    cfg = json.load(f)

VIDEO_PATH   = cfg['video_path']
RESULTS_DIR  = cfg['results_dir']
PATCH_SIZE   = cfg['patch_size']      # e.g. 50
OUTPUT_VIDEO = os.path.join(RESULTS_DIR, 'hough_prototype.mp4')

os.makedirs(RESULTS_DIR, exist_ok=True)

# 2) Init MediaPipe Hands
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3) Open input video + prepare writer
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# We will stack the grid below the frame, so output height = h + grid_h
# Each patch is 2*PATCH_SIZE square; we arrange up to 5 patches per row
patch_display = 2 * PATCH_SIZE
ncols = 5
# Maximum of 5 patches per frame, so grid rows = 1 always—but code handles more
grid_rows = 1
grid_h    = grid_rows * patch_display
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h + grid_h)
)
print(f"[INFO] Writing prototype video to: {OUTPUT_VIDEO}")

def make_grid(patches, pw, ph, ncols=5):
    """Assemble list of patches into a grid image."""
    rows = (len(patches) + ncols - 1) // ncols
    grid = np.zeros((ph * rows, pw * ncols, 3), dtype=np.uint8)
    for idx, patch in enumerate(patches):
        r, c = divmod(idx, ncols)
        # resize patch to exactly (ph, pw)
        resized = cv2.resize(patch, (pw, ph))
        grid[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = resized
    return grid

# 4) Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    patches = []
    markers = []  # (x, y, has_circle)

    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            # draw landmarks for reference (optional)
            # mp.solutions.drawing_utils.draw_landmarks(frame, handlm)
            hgt, wid, _ = frame.shape
            # fingertip indices: [4,8,12,16,20]
            for tip_idx in [4, 8, 12, 16, 20]:
                lm = handlm.landmark[tip_idx]
                cx = int(lm.x * wid)
                cy = int(lm.y * hgt)

                # crop patch around (cx, cy)
                r = PATCH_SIZE
                x0, y0 = max(0, cx - r), max(0, cy - r)
                x1, y1 = min(wid, cx + r), min(hgt, cy + r)
                patch = frame[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                patches.append(patch)

                # detect circles
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=PATCH_SIZE,
                    param1=100,
                    param2=30,
                    minRadius=5,
                    maxRadius=PATCH_SIZE//2
                )
                has_ring = circles is not None
                markers.append((cx, cy, has_ring))

    # overlay circle markers onto frame
    for (x, y, ok) in markers:
        color = (0,255,0) if ok else (0,0,255)
        cv2.circle(frame, (x,y), PATCH_SIZE//2 if ok else 12, color, thickness=3)

    # build grid of patches
    grid = make_grid(patches, patch_display, patch_display, ncols)

    # stack frame + grid vertically
    # if grid height < grid_h, pad it
    if grid.shape[0] < grid_h:
        pad = np.zeros((grid_h - grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        grid = np.vstack((grid, pad))
    combined = np.vstack((frame, cv2.resize(grid, (w, grid_h))))

    # write to output
    out.write(combined)

cap.release()
out.release()
print("[INFO] Prototype video complete.")
