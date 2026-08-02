#!/bin/bash
#
# Download the TUM RGB-D scenes from the ORB-SLAM2 evaluation (Mur-Artal &
# Tardos, T-RO 2017, Table 3) into datasets/, and write the calibration.yaml
# each scene needs (per-freiburg Kinect intrinsics from the TUM website).
#
# Usage:
#   ./download_tum_rgbd.sh          # all scenes, skipping ones already present
set -euo pipefail

DATASETS_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_URL="https://cvg.cit.tum.de/rgbd/dataset"

SCENES=(
  freiburg1/rgbd_dataset_freiburg1_desk
  freiburg1/rgbd_dataset_freiburg1_desk2
  freiburg1/rgbd_dataset_freiburg1_room
  freiburg2/rgbd_dataset_freiburg2_desk
  freiburg2/rgbd_dataset_freiburg2_xyz
  freiburg3/rgbd_dataset_freiburg3_long_office_household
  freiburg3/rgbd_dataset_freiburg3_nostructure_texture_near_withloop
)

# Kinect intrinsics per camera (fr3 ships undistorted).
calibration() {
  case "$1" in
    freiburg1) cat <<EOF
# oST version 5.0 parameters

width: 640
height: 480

depth_scale: 5000.0

fx: 517.3
fy: 516.5
cx: 318.6
cy: 255.3

distortion: [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
EOF
    ;;
    freiburg2) cat <<EOF
# oST version 5.0 parameters

width: 640
height: 480

depth_scale: 5000.0

fx: 520.9
fy: 521.0
cx: 325.1
cy: 249.7

distortion: [0.2312, -0.7849, -0.0033, -0.0001, 0.9172]
EOF
    ;;
    freiburg3) cat <<EOF
# oST version 5.0 parameters

width: 640
height: 480

depth_scale: 5000.0

fx: 535.4
fy: 539.2
cx: 320.1
cy: 247.6

distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
EOF
    ;;
  esac
}

mkdir -p "$DATASETS_DIR"
for scene in "${SCENES[@]}"; do
  group="${scene%%/*}"
  name="${scene##*/}"
  dir="$DATASETS_DIR/$name"
  if [ -d "$dir" ]; then
    echo "skip $name (already present)"
    continue
  fi
  echo "downloading $name"
  wget -c -q --show-progress "$BASE_URL/$scene.tgz" -O "$DATASETS_DIR/$name.tgz"
  tar -xzf "$DATASETS_DIR/$name.tgz" -C "$DATASETS_DIR"
  rm "$DATASETS_DIR/$name.tgz"
  calibration "$group" > "$dir/calibration.yaml"
done
