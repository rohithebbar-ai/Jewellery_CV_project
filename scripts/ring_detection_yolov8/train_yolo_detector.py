"""# train.py

import os
from random import triangular
import time
import json
import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


def load_config(path: str) -> dict:
    #Load JSON config from disk.
    with open(path, 'r') as f:
        return json.load(f)

def train_yolo(cfg: dict) -> Path:

    Fine-tune YOLOv8 according to cfg.
    Returns the path to the run directory (where metrics.csv lives).
  
    # Initialize model from pretrained weights
    model = YOLO(cfg['model_name'])  
    start = time.time()

    # Train
    model.train(
        data=cfg['data_yaml'],
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=cfg['batch'],
        project=cfg['project'],
        name=cfg['name'],
        augment = True,
        auto_augment='RandAugment', # strong photometric/geometric
        mosaic=1.0,                 # enable mosaic mixing
        mixup=0.5,  
        # freeze=cfg.get('freeze_layers', None)  # uncomment if you want to freeze early layers
    )

    elapsed_min = (time.time() - start) / 60
    print(f"[INFO] Training completed in {elapsed_min:.1f} minutes")

    # Return the directory where Ultralytics dumped metrics.csv
    return Path(cfg['project']) / cfg['name']

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on ring dataset and log to TensorBoard"
    )
    parser.add_argument(
        '--config', type=str, default='config.json',
        help="Path to your JSON config file"
    )
    args = parser.parse_args()

    # Load settings
    cfg = load_config(args.config)
    os.makedirs(cfg['tensorboard_logdir'], exist_ok=True)

    # 1) Train
    run_dir = train_yolo(cfg)

    # 2) Log metrics
    metrics_csv = run_dir / 'metrics.csv'
    if not metrics_csv.exists():
        metrics_csv = run_dir / 'results.csv'
    
    print(f"About to log", metrics_csv)

    log_to_tensorboard(cfg['tensorboard_logdir'], metrics_csv)

if __name__ == "__main__":
    main()"""



# train.py

import os
import time
import json
import argparse
from pathlib import Path
from ultralytics import YOLO

def load_config(path: str) -> dict:
    """Load JSON config from disk."""
    with open(path, 'r') as f:
        return json.load(f)

def train_yolo(cfg: dict) -> Path:
    """
    Fine-tune YOLOv8 according to cfg.
    Returns the path to the run directory.
    """
    model = YOLO(cfg['model_name'])
    start = time.time()

    model.train(
        data=cfg['data_yaml'],
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=cfg['batch'],
        project=cfg['project'],
        name=cfg['name'],
        augment=True,
        auto_augment='RandAugment',
        mosaic=1.0,
        mixup=0.5,
        # no freeze → full fine-tune
    )

    elapsed = (time.time() - start) / 60
    print(f"[INFO] Training completed in {elapsed:.1f} min")

    run_dir = Path(cfg['project']) / cfg['name']
    print(f"[INFO] Artifacts saved to: {run_dir.resolve()}")
    return run_dir

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on ring dataset"
    )
    parser.add_argument(
        '--config', type=str, default='config.json',
        help="Path to JSON config"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg['project'], exist_ok=True)

    train_yolo(cfg)

if __name__ == "__main__":
    main()
