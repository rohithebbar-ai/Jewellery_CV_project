# Finger & Jewellery (💍 ) Tracking and Localization.

## Overview
This project tracks rings on fingers in real-world videos using a combination of YOLOv8 for object detection, DeepSORT for multi-object tracking, and MediaPipe Hands for finger landmark localization.

Key scripts for this pipeline:
- **MediaPipe Landmark Module** (`scripts/hand_landmarker.py`): Detects 21 hand landmarks in each frame and identifies the ring finger joints.
- **YOLOv8 Ring Detector** (`scripts/train_yolo_detector.py` / `models/ring_detector/`): Fine-tuned on rings-on-hand images to detect ring instances within cropped finger regions. 
- **Video Inference** (`scripts/video_inference.py`): Runs the combined pipeline on demo videos, draws a translucent mask + thick bounding box + confidence label, and writes an annotated output file.

### 📌 Key Features
- **YOLOv8** for detecting jewelry (rings) from RGB frames
- **DeepSORT** for assigning consistent track IDs across video frames
- **MediaPipe** for extracting 3D finger landmarks
- **Finger association** for identifying which finger a ring is worn on
- **CSV Logging** for detailed tracking info (frame, track ID, bounding box, confidence, finger name)
- **Metrics Evaluation** on test video for detection, tracking, association, and stability.

## Repository Structure
```
Jewellery_CV_project/

├── scripts/          # core pipeline scripts
|   |── mediapipe_hand_detection/
│   |   ├── hand_landmarker.py
|   |   ├── extract_frames.py
|   |   ├── hough_prototype.py
│   |   ├── main.py
|   |   ├── ring_candidates.py
|   └── ring_detection_yolov8/
│       ├── convert_all_labels_to_yolo.py
|       ├── test_label_annotation.py
│       ├── train_yolo_detector.py
|       ├── video_inference.py
|       ├── via_to_gt.py
|       └── utils/
│           ├── drawing_utils.py
│           ├── mediapipe_utils.py
│           ├── metrics.py
|           ├── ring_finger_matcher.py
├── YoloV8_Results/              # trained model weights and configs
│   └── ring_detector/
│       ├── weights/best.pt
│       └── labels.jpg   # plotted label distribution
│       └── F1_curve.jpg
│       └── PR_curve.jpg
|-- output_anna_demo_video_1.csv
│── Design_Report_Rohit_Hebbar.pdf
│── processed_gt.pkl
│── config_mediapipe.json # JSON/ YAML config files
|__ config_yolo.json
├── README.md            # this file
└── requirements.txt     # pip install dependencies
```

The input video of Anna wearing two rings is also taken from open source website (pexels). The link to this can be found here.
https://drive.google.com/drive/folders/1K7d8S23CNhajPP5urI9vxSbaihlwNwh2?usp=sharing

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
Make sure you have the following:
- OpenCV
- Ultralytics (YOLOv8)
- mediapipe
- deep_sort_realtime
- motmetrics
- numpy, matplotlib, tqdm, pandas
  
3. **Download MediaPipe models** (if needed) and place at `venv/lib/python*/site-packages/mediapipe/models/`.

4. If you want to use the dataset for finetuning yolo for ring detection then you can download it on this link. All the data is taken from open source and manually labelled using labellmg.
 https://drive.google.com/drive/folders/1YCnSLYkh-kb_fMOOBuTN2BkdPO3fovNr?usp=sharing

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


## ▶️ How to Run Inference

### Step 1: Run YOLOv8 + DeepSORT + MediaPipe Inference

```bash
python3 scripts/ring_detection_yolov8/video_inference.py   --model models/ring_detector/weights/best.pt   --source data/anna_demo.mp4   --conf 0.30 --iou 0.30   --out results/annotated_output.mp4   --csv results/predictions.csv
```

This will:
- Save the annotated video to `results/annotated_output.mp4`
- Save tracking + finger association logs to `results/predictions.csv`

---

## 📊 Metrics Evaluation

For this video i converted them into frames and there were 1210 frames, however annotating every frame was a tedious task and it has to be manual annotation so i took random unbiased 50 frames using the script 'ring_detection_yolov8/test_label_annotation.py'

### Step 2: Convert VIA annotations to GT format

```bash
python3 scripts/ring_detection_yolov8/via_to_gt.py   --via results/your_via_export.json   --out results/gt.pkl
```

### Step 3: Evaluate Metrics

```bash
python3 scripts/ring_detection_yolov8/video_inference.py   --model models/ring_detector/weights/best.pt   --source data/anna_demo.mp4   --csv results/predictions.csv   --gt results/gt.pkl
```

---

## 🧠 Why YOLOv8 + MediaPipe + DeepSORT?

- YOLOv8 is fast and efficient for object detection.
- MediaPipe gives reliable finger landmark localization.
- Together, they allow us to associate rings with specific fingers.
- DeepSORT maintains track IDs across frames.

**Without MediaPipe**, we would not know which finger the ring is on, only where it is spatially.

---

## Experiments & Decision Log
- **Approaches tried**:
  - Mediapipe + Hough transform (prototype) - didn't work well.   
  - MediaPipe → YOLO crop pipeline (current) ✔️
  - Mask-RCNN segmentation head (future work)
  - CAD overlay via PyTorch3D (not enough time)

- **Data & Augmentations**:
  - ~50 ring-on-hand images hand-annotated
  - RandAugment + MixUp + Mosaic improved recall by ~10%
  - Hard negatives (empty-hand, bracelets only) reduced false positives

- **Results**:
  - mAP@0.5: **0.77** on held-out set
  - Visual outputs: `results/metrics_plots.png`, `results/val_batch_pred.jpg`
  - The output data and results from yolov8 can be found here. 
https://drive.google.com/drive/folders/1obQbFBas9fBcFqtzuC62j3nWm8QUYwjf?usp=sharing

The metrics on inference video is :
| Category       | Metric                     | Value       |
|----------------|----------------------------|-------------|
| Detection      | precision                  | 0.250000    |
| Detection      | recall                     | 0.263158    |
| Detection      | f1                         | 0.256410    |
| Detection      | per_frame_detection_rate   | 0.130435    |
| IoU stats      | mean_iou                   | 0.244541    |
| IoU stats      | median_iou                 | 0.062445    |
| Assoc Accuracy | assoc_accuracy             | 0.000000    |
| Tracking       | mota                       | 0.157895    |
| Tracking       | idf1                       | 0.217949    |
| Tracking       | num_switches               | 20.000000   |
| Tracking       | mostly_tracked             | 1.000000    |
| Tracking       | mostly_lost                | 1.000000    |
| Stability      | max_drift (ID 1)           | 626.516161  |
| Stability      | area_var_norm (ID 1)       | 0.518888    |
| Stability      | angle_std (ID 1)           | 0.693014    |
| Stability      | max_drift (ID 2)           | 832.358368  |
| Stability      | area_var_norm (ID 2)       | 0.553418    |
| Stability      | angle_std (ID 2)           | 0.892142    |
| Stability      | max_drift (ID -1)          | 660.713061  |
| Stability      | area_var_norm (ID -1)      | 1.000000    |
| Stability      | angle_std (ID -1)          | 0.000000    |

<p float="left">
  <img src="assets/hand_ring_1.png" width="45%" />
  <img src="assets/hand_ring_2.png" width="45%" />
</p>

<p float="left">
  <em style="margin-right: 40%;">Ring on left thumb</em>
  <em>Ring on right middle finger</em>
</p>

## ⚠️ Limitations 

- MediaPipe fails in cases of occlusion or motion blur
- Only YOLO bounding boxes used; segmentation masks not trained
- Low Detection Precision and Recall:
The detection module achieved only ~25% precision and ~26% recall, indicating that many rings are either missed or falsely detected. This could be due to limited training data, challenging lighting conditions, occlusions, or small ring sizes in the input frames.
- Poor Association Accuracy:
The association module shows an accuracy of 0.0, suggesting it is currently unable to reliably link detected rings to specific fingers across frames, which impacts consistent tracking.
- Weak Tracking Performance:
With a MOTA of ~15.8% and IDF1 of ~21.8%, the tracking pipeline struggles to maintain consistent identities, likely due to frequent ID switches (20 total) and ambiguous finger-ring assignments in cluttered or fast-moving scenes.
- Unstable Localization (Stability Metrics):
High drift values (e.g., 832 px max drift for ID 2) and varying area/angle consistency indicate instability in ring localization, especially across time. This makes it less reliable for long or real-time sequences.
- Low IoU Scores:
The mean IoU of ~24% and median IoU of ~6% suggest a mismatch between predicted and ground-truth ring regions. This reflects poor spatial alignment, which may stem from inaccurate bounding box predictions or temporal inconsistencies.


🔧 Future Work
- Improve Ring Detection Accuracy:
Augment the training dataset with more diverse hand poses, lighting conditions, and ring styles. Use techniques like data augmentation, synthetic data generation, or transfer learning from larger object detection models.
- Enhance Finger-Ring Association Logic:
Incorporate temporal context, such as tracking hand landmarks across frames, or introduce graph-based matching algorithms to better associate rings with specific fingers consistently.
- Refine Tracking Pipeline:
Explore advanced trackers (e.g., ByteTrack, DeepSORT with re-ID) or fine-tune tracking heuristics to reduce ID switches and maintain identity persistence over time.
- Increase Localization Stability:
Apply temporal smoothing filters (e.g., Kalman filter or exponential moving averages) on the ring position and orientation to reduce drift and jitter in dynamic scenes.
- Optimize Bounding Box Alignment:
Improve post-processing using IoU-based refinements or regression heads to better match predicted boxes with true ring shapes, possibly leveraging segmentation masks instead of just bounding boxes.
- Real-Time Evaluation:
Profile runtime performance and explore model quantization or ONNX/TensorRT conversion for real-time inference on edge devices or embedded platform.

Could also try:
   - SAHI for small-object boosting
   - Add instance segmentation for pixel-precise ring masks
   - YOLOv8-seg with re-annotated mask labels
   - Replace MediaPipe with 3D hand pose model (e.g. FrankMocap)
   - Evaluate on longer videos and more lighting conditions

---
*Author: Rohit Hebbar*  
*Date: 25-04-2025*

