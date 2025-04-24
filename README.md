# Virtual Ring Tracking on Anna’s Hand

## Overview
This repository implements a two-stage pipeline to detect and track rings on Anna’s fingers in video footage. It combines MediaPipe hand landmark estimation with a fine-tuned YOLOv8 detector, cropping per-finger regions for robust ring localization.

### Key Components
- **MediaPipe Landmark Module** (`scripts/hand_landmarker.py`): Detects 21 hand landmarks in each frame and identifies the ring finger joints.
- **YOLOv8 Ring Detector** (`scripts/train.py` / `models/ring_detector/`): Fine-tuned on ring-on-hand images to detect ring instances within cropped finger regions.
- **Video Inference** (`scripts/video_inference.py`): Runs the combined pipeline on demo videos, draws a translucent mask + thick bounding box + confidence label, and writes an annotated output file.

## Repository Structure
```
Jewellery_CV_project/

├── scripts/          # core pipeline scripts
|   └── mediapipe_hand_detection/
│       ├── hand_landmarker.py
│       ├── main.py
|       ├── ring_candidates.py
|   └── ring_detection_yolov8/
│       ├── convert_all_labels_to_yolo.py
│       ├── train.py
|       ├── video_inference.py
|       └── utils/
│           ├── drawing_utils.py
│           ├── mediapipe_utils.py
|           ├── ring_finger_matcher.py
├── models/              # trained model weights and configs
│   └── ring_detector/
│       ├── weights/best.pt
│       └── labels.jpg   # plotted label distribution
├         
│── config_mediapipe.json # JSON/ YAML config files
│- config_yolo.json
├── README.md            # this file
└── requirements.txt     # pip install dependencies
```

## Setup & Installation
1. **Clone** this repository:
   ```bash
   git clone https://github.com/yourusername/Jewellery_CV_project.git
   cd Jewellery_CV_project
   ```
2. **Create a virtual environment** and install requirements:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Download MediaPipe models** (if needed) and place at `venv/lib/python*/site-packages/mediapipe/models/`.

## Running the Pipeline

### 1) Extract Frames (optional)
```bash
python scripts/hand_landmarker.py \
  --source data/anna_demo.mp4 \
  --out_dir data/frames/
```

### 2) Train the YOLOv8 Ring Detector
```bash
python scripts/train.py --config configs/config_yolo.json
```
- Model weights and training metrics are saved under `models/ring_detector/`.

### 3) Inference on Video
```bash
python scripts/video_inference.py \
  --model models/ring_detector/weights/best.pt \
  --source data/anna_demo.mp4 \
  --out outputs/anna_demo_annotated.mp4
```

## Experiments & Decision Log
- **Approaches tried**:
  - Direct YOLOv8 on full frames (high false positive on bracelets) 🗙
  - MediaPipe → YOLO crop pipeline (current) ✔️
  - Mask-RCNN segmentation head (future work)
  - CAD overlay via PyTorch3D (not enough time)

- **Data & Augmentations**:
  - ~150 ring-on-hand images hand-annotated
  - RandAugment + MixUp + Mosaic improved recall by ~10%
  - Hard negatives (empty-hand, bracelets only) reduced false positives

- **Results**:
  - mAP@0.5: **0.77** on held-out set
  - Average center-jitter < 3 px over 100 frames
  - Visual outputs: `results/metrics_plots.png`, `results/val_batch_pred.jpg`

## Next Steps
- Add **instance segmentation** for pixel-precise ring masks
- Integrate **3D CAD overlays** matching hand pose
- Evaluate on longer videos and more lighting conditions

---
*Author: Rohit Hebbar*  
*Date: 25-04-2025*

