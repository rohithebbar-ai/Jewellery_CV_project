#!/usr/bin/env python3
"""
sample_frames.py

Extracts N frames from a video in a uniform, unbiased way by splitting
the total frame count into N equal intervals and randomly sampling one
frame from each interval.
"""

import cv2
import os
import random
import argparse

def sample_frames(video_path: str, out_dir: str, n_frames:int=50, seed: int=42):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Could not determine total frame count.")

    # Compute the interval size
    interval = total_frames / n_frames

    # Prepare output
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    # Pick one random frame index per interval
    frame_indices = []
    for i in range(n_frames):
        start = int(i * interval)
        end = int((i + 1) * interval) - 1
        if end < start:
            idx = start
        else:
            idx = random.randint(start, end)
        frame_indices.append(idx)

    # Extract and save
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame {idx}")
            continue
        # Name by original frame number
        frame_no = idx + 1
        out_path = os.path.join(out_dir, f"frame_{frame_no:06d}.jpg")
        cv2.imwrite(out_path, frame)

    cap.release()
    print(f"Extracted {len(frame_indices)} frames to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample N unbiased frames from a video"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--out_dir", "-o",
        default = "frames50_testlabels",
        help = "Directory to save sampled frames"
    )
    parser.add_argument(
        "--num",        "-n",
        type=int,
        default=50,
        help="Number of frames to sample"
    )
    parser.add_argument(
        "--seed",       "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    sample_frames(args.video, args.out_dir, args.num, args.seed)
