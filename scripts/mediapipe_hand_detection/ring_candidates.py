import os
import cv2 
import json
import numpy as np 

#Load config
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, 'config_mediapipe.json')
with open(CONFIG_PATH, 'r') as f:
    cfg = json.load(f)


FRAMES_DIR =  cfg['frames_dir']
PATCHES_DIR =  cfg['patches_dir']
LANDMARKS_FILE = cfg['landmarks_file']
PATCH_SIZE = cfg['patch_size']

class RingCandidateExtractor:
    def __init__(self):
        os.makedirs(PATCHES_DIR, exist_ok=True)
        with open(LANDMARKS_FILE) as f:
            self.landmark_records = json.load(f)
            
    def crop_detect(self):
        ring_fingers = {"Left":[16], "Right":[16]}
        for rec in self.landmark_records:
            for idx in ring_fingers[rec["hand"]]:
                frame_idx = rec['frame']
                hand_label = rec['hand']
                frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.jpg")
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                
                h, w = frame.shape[:2]
                for idx in [16]:
                    x_pct, y_pct, _ = rec['landmarks'][idx]
                    cx, cy = int(x_pct * w), int(y_pct * w)
                    x1 = max(cx - PATCH_SIZE, 0)
                    y1 = max(cy - PATCH_SIZE, 0)
                    x2 = max(cx + PATCH_SIZE, w)
                    y2 = min(cy + PATCH_SIZE, h)
                    patch = frame[y1:y2, x1:x2]
                    patch_name = f'frame{frame_idx:04d}_pt{idx}_{hand_label}.png'
                    cv2.imwrite(os.path.join(PATCHES_DIR, patch_name), patch)
                    self._detect_circles(patch, frame_idx, idx, hand_label)
                
    def _detect_circles(self, patch, frame_idx, idx, hand_label):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()
        if mean_val < 100:
            return # too dark to be a shiny ring
        print(f"Mean_val value :", mean_val)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1, minDist=PATCH_SIZE,
            param1=100, param2=50,
            minRadius=5, maxRadius=PATCH_SIZE//3
        )
        if circles is not None:
            x, y, r = max(circles[0], key=lambda c: c[2])
            
            if 8 < r < PATCH_SIZE//3:
                print(f"[FOUND] frame {frame_idx}: ring r={r}")
                print(f"[FOUND] Ring-like circle in frame {frame_idx}, fingertip {idx}, hand {hand_label}")
