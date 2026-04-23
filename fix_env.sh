#!/bin/bash
# fix_env.sh  —  AutoSense dependency fix (macOS Apple Silicon, Python 3.11)
#
# THE REAL SITUATION (confirmed from mediapipe GitHub issues):
#   mediapipe >=0.10.13  needs  protobuf >=4.25.3, <5
#   tensorflow 2.21.0    needs  protobuf >=6.31.1
#   These are PERMANENTLY incompatible. No version satisfies both.
#
# SOLUTION: remove tensorflow from this environment.
#   Your project (main.py, detector.py, risk_classifier.py) does NOT import
#   tensorflow anywhere. It was only installed as a transitive dep and is
#   the entire cause of the protobuf war.
#
# Run:
#   conda deactivate          <-- IMPORTANT: must not be active
#   source venv311/bin/activate
#   chmod +x fix_env.sh && ./fix_env.sh

set -e

echo "=== Step 1: remove everything conflicting ==="
pip uninstall -y \
    tensorflow tensorflow-macos tensorflow-metal \
    mediapipe protobuf \
    opencv-python opencv-contrib-python opencv-python-headless \
    numpy 2>/dev/null || true

echo ""
echo "=== Step 2: install clean compatible stack ==="
# mediapipe 0.10.18 is stable, needs protobuf >=4.25.3,<5
# opencv-python-headless avoids SDL2 dylib conflict with pygame
# numpy <2.3 keeps scipy happy
pip install \
    "protobuf>=4.25.3,<5" \
    "mediapipe==0.10.18" \
    "opencv-python-headless==4.9.0.80" \
    "numpy>=1.23.5,<2.3.0"

echo ""
echo "=== Step 3: verify everything works ==="
python - <<'EOF'
print("Testing imports...")

import mediapipe as mp
test = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=False,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)
test.close()
print("[OK] MediaPipe FaceMesh — working!")

import cv2
print(f"[OK] OpenCV {cv2.__version__}")

import numpy as np
print(f"[OK] NumPy {np.__version__}")

import scipy
print(f"[OK] SciPy {scipy.__version__}")

import google.protobuf
print(f"[OK] protobuf {google.protobuf.__version__}")

print("")
print("All good — run: python main.py")
EOF
