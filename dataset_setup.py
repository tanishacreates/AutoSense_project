"""
dataset_setup.py
----------------
Dataset acquisition and preprocessing for the driver assistance project.

Datasets used (shared responsibility between Person 1 and Person 2):
  1. CEW   (Closed Eyes in the Wild) — eye state classification
  2. YawDD (Yawning Detection Dataset) — yawning videos
  3. NTHU-DDD (Driver Drowsiness Detection Dataset) — full drowsiness videos

This script:
  - Provides download instructions for each dataset (most require registration)
  - Generates a synthetic tabular dataset for Person 2's ML training
    (no download needed — uses the feature distributions from published papers)
  - Converts raw video datasets to frame CSVs for ML training

Usage:
    python dataset_setup.py --generate-synthetic   # creates data/synthetic_train.csv
    python dataset_setup.py --extract-frames PATH  # extracts frames from a video dataset
    python dataset_setup.py --info                 # print dataset information
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  Dataset information                                                  #
# ------------------------------------------------------------------ #

DATASET_INFO = {
    "CEW": {
        "full_name"   : "Closed Eyes in the Wild",
        "description" : "84,898 images of open/closed eyes for binary classification.",
        "url"         : "http://parnec.nuaa.edu.cn/liyang/research/closed_eyes_in_the_wild.html",
        "size"        : "~480 MB",
        "person"      : "Person 1 (CNN training for eye state)",
        "format"      : "Images — closed/ and open/ directories",
        "classes"     : ["open", "closed"],
    },
    "YawDD": {
        "full_name"   : "Yawning Detection Dataset",
        "description" : "101 driving videos with yawning annotations.",
        "url"         : "https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset",
        "size"        : "~4.2 GB",
        "person"      : "Both (vision training + integration testing)",
        "format"      : "AVI videos organised by driver ID",
        "classes"     : ["no_yawn", "yawn"],
    },
    "NTHU-DDD": {
        "full_name"   : "NTHU Driver Drowsiness Detection Dataset",
        "description" : "36 subjects, various lighting + glasses conditions.",
        "url"         : "http://nthu-datalab.cs.nthu.edu.tw/lab/projects/Driver_Drowsiness_Detection/Driver_Drowsiness_Detection.html",
        "size"        : "~8.2 GB",
        "person"      : "Both (primary benchmark dataset)",
        "format"      : "MP4 videos, CSV labels",
        "classes"     : ["alert", "low vigilance", "drowsy"],
    },
}


def print_dataset_info():
    """Print dataset descriptions and download instructions."""
    print("\n" + "="*60)
    print("  DRIVER ASSISTANCE — DATASET REFERENCE GUIDE")
    print("="*60)
    for name, info in DATASET_INFO.items():
        print(f"\n📦 {name} — {info['full_name']}")
        print(f"   {info['description']}")
        print(f"   URL        : {info['url']}")
        print(f"   Size       : {info['size']}")
        print(f"   Assigned to: {info['person']}")
        print(f"   Format     : {info['format']}")
        print(f"   Classes    : {info['classes']}")
    print()
    print("NOTES:")
    print("  - CEW and NTHU-DDD require registration for download.")
    print("  - YawDD is openly available on IEEE DataPort.")
    print("  - Person 2 can train the risk classifier on the synthetic")
    print("    dataset (see --generate-synthetic) while waiting for")
    print("    Person 1 to process the real datasets.")
    print("="*60 + "\n")


# ------------------------------------------------------------------ #
#  Synthetic tabular dataset generation                                 #
# ------------------------------------------------------------------ #

def generate_synthetic_csv(
    output_path: str = None,
    n_samples:   int = 8000,
    seed:        int = 42,
) -> str:
    """
    Generate a realistic tabular dataset based on published feature
    statistics from NTHU-DDD research papers.

    Columns match the fusion_engine.VisionFeatures and
    sensor_module.SensorReading field names, plus a 'label' column:
      0 = alert  |  1 = warning  |  2 = drowsy

    Returns the output CSV path.
    """
    from risk_classifier import generate_synthetic_dataset, FEATURE_NAMES

    if output_path is None:
        output_path = str(DATA_DIR / "synthetic_train.csv")

    logger.info(f"Generating {n_samples} synthetic samples ...")
    X, y = generate_synthetic_dataset(n_samples=n_samples, seed=seed)

    label_names = {0: "alert", 1: "warning", 2: "drowsy"}
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"]      = y
    df["label_name"] = df["label"].map(label_names)

    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic dataset saved: {output_path}  ({len(df)} rows)")

    # Print class distribution
    print(f"\nDataset saved: {output_path}")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    for label, name in label_names.items():
        count = (df["label"] == label).sum()
        pct   = 100 * count / len(df)
        print(f"  {name:<10}: {count:5d}  ({pct:.1f}%)")

    return output_path


# ------------------------------------------------------------------ #
#  Frame extractor for real video datasets                             #
# ------------------------------------------------------------------ #

def extract_frames_from_video(
    video_path:  str,
    output_dir:  str,
    label:       str = "unknown",
    sample_fps:  float = 5.0,
    max_frames:  int = 500,
) -> int:
    """
    Extract sampled frames from a driver video file.
    Saves frames as:  output_dir/<label>/<basename>_frame_NNNN.jpg

    Returns number of frames saved.
    """
    try:
        import cv2
    except ImportError:
        print("opencv-python required: pip install opencv-python")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return 0

    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_skip   = max(1, int(video_fps / sample_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(output_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    basename   = Path(video_path).stem
    saved      = 0
    frame_idx  = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            fname = out_dir / f"{basename}_frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {saved}/{total_frames} frames → {out_dir}")
    return saved


def batch_extract_nthu(dataset_root: str, output_dir: str = "data/frames"):
    """
    Batch extract frames from NTHU-DDD dataset structure:
      dataset_root/
        Training_Evaluation_Dataset/
          Training/
            001/
              drowsy.avi
              non-drowsy.avi
    """
    root = Path(dataset_root)
    total = 0
    for video_file in root.rglob("*.avi"):
        label = "drowsy" if "drowsy" in video_file.name.lower() else "alert"
        saved = extract_frames_from_video(
            video_path  = str(video_file),
            output_dir  = output_dir,
            label       = label,
            sample_fps  = 3.0,
            max_frames  = 300,
        )
        total += saved
    print(f"\nTotal frames extracted from NTHU-DDD: {total}")
    return total


# ------------------------------------------------------------------ #
#  CLI                                                                  #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Dataset setup utility")
    p.add_argument("--info",             action="store_true",
                   help="Print dataset download information")
    p.add_argument("--generate-synthetic", action="store_true",
                   help="Generate synthetic training CSV")
    p.add_argument("--n-samples", type=int, default=8000,
                   help="Number of synthetic samples (default: 8000)")
    p.add_argument("--extract-frames", metavar="VIDEO_PATH",
                   help="Extract frames from a single video file")
    p.add_argument("--extract-nthu", metavar="DATASET_ROOT",
                   help="Batch extract frames from NTHU-DDD dataset")
    p.add_argument("--label", default="unknown",
                   help="Label for extracted frames (alert/drowsy/distracted)")
    p.add_argument("--output-dir", default="data/frames",
                   help="Output directory for extracted frames")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.info:
        print_dataset_info()

    if args.generate_synthetic:
        generate_synthetic_csv(n_samples=args.n_samples)

    if args.extract_frames:
        extract_frames_from_video(
            video_path = args.extract_frames,
            output_dir = args.output_dir,
            label      = args.label,
        )

    if args.extract_nthu:
        batch_extract_nthu(args.extract_nthu, args.output_dir)

    if not any([args.info, args.generate_synthetic,
                args.extract_frames, args.extract_nthu]):
        print("No action specified. Run with --help for options.")
        print_dataset_info()


if __name__ == "__main__":
    main()
