# scripts/extract_frames.py
import os
import cv2
import json

# load your config
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, 'config_mediapipe.json')
with open(CONFIG_PATH, 'r') as f:
    cfg = json.load(f)


VIDEO_PATH  = cfg['video_path']
FRAMES_DIR  = cfg['frames_dir']
os.makedirs(FRAMES_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fname = os.path.join(FRAMES_DIR, f"frame_{idx:04d}.jpg")
    cv2.imwrite(fname, frame)
    idx += 1

cap.release()
print(f"Extracted {idx} frames into {FRAMES_DIR}")
