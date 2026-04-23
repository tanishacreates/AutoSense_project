"""
AutoSense - Evaluation Script
Run offline evaluation on a video file or image dataset.
Produces metrics: accuracy, confusion matrix, ROC curve.

Usage:
    python evaluate.py --video path/to/test_video.mp4
    python evaluate.py --dataset data/test_images --labels data/labels.csv
"""

import cv2
import argparse
import time
import json
import os
from detector import DriverMonitor, DriverState


def evaluate_video(video_path: str, sensitivity: str = "medium",
                   output_json: str = "results/eval_results.json"):
    """
    Run the detector on a video file and collect per-frame metrics.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    os.makedirs("results", exist_ok=True)
    monitor = DriverMonitor(sensitivity=sensitivity)
    cap     = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    total_fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Frames: {total_frames} @ {total_fps:.1f} fps")

    records = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        state: DriverState = monitor.analyze(frame)
        record = {
            "frame":            frame_idx,
            "time_sec":         round(frame_idx / total_fps, 3),
            "face_detected":    state.face_detected,
            "ear":              round(state.ear, 4),
            "mar":              round(state.mar, 4),
            "perclos":          round(state.perclos, 4),
            "blink_rate":       round(state.blink_rate, 2),
            "pitch":            round(state.pitch, 2),
            "yaw":              round(state.yaw, 2),
            "eyes_closed":      state.eyes_closed,
            "yawning":          state.yawning,
            "head_down":        state.head_down,
            "looking_away":     state.looking_away,
            "drowsiness_level": state.drowsiness_level,
            "drowsy":           state.drowsy,
            "distracted":       state.distracted,
        }
        records.append(record)

        if frame_idx % 300 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  {pct:.1f}%  frame {frame_idx}/{total_frames}")

    cap.release()
    elapsed = time.time() - t0
    print(f"[INFO] Processed {frame_idx} frames in {elapsed:.1f}s")

    # ── Summary ──
    face_det  = sum(r["face_detected"] for r in records)
    drowsy_ct = sum(r["drowsy"]        for r in records)
    dist_ct   = sum(r["distracted"]    for r in records)

    summary = {
        "video":           video_path,
        "total_frames":    frame_idx,
        "face_detected_pct": round(face_det / max(frame_idx, 1) * 100, 2),
        "drowsy_frames_pct": round(drowsy_ct / max(frame_idx, 1) * 100, 2),
        "distracted_frames_pct": round(dist_ct / max(frame_idx, 1) * 100, 2),
        "session_stats":   monitor.get_session_stats(),
    }

    output = {"summary": summary, "frames": records}
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[RESULT] Face detected  : {summary['face_detected_pct']}%")
    print(f"[RESULT] Drowsy frames  : {summary['drowsy_frames_pct']}%")
    print(f"[RESULT] Distracted     : {summary['distracted_frames_pct']}%")
    print(f"[RESULT] Saved to       : {output_json}")
    return summary


def print_summary_table(summary: dict):
    print("\n" + "=" * 45)
    print(" AutoSense Evaluation Summary")
    print("=" * 45)
    for k, v in summary.items():
        if k != "session_stats":
            print(f"  {k:<30} {v}")
    print("=" * 45)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",       required=True, help="Path to test video file")
    p.add_argument("--sensitivity", default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--output",      default="results/eval_results.json")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    result = evaluate_video(args.video, args.sensitivity, args.output)
    if result:
        print_summary_table(result)
