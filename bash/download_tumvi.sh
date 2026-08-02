#!/bin/bash
#
# Download the TUM-VI room sequences from the ORB-SLAM3 evaluation (Campos et
# al., T-RO 2021) into datasets/tumvi/ (euroc-format 512x512 export). The room
# sequences are the ones with full mocap ground truth.
#
# The remaining environments (corridor, magistrale, slides, outdoors) only
# have ground truth at start/end and total hundreds of GB; add them to
# SEQUENCES if needed.
#
# Usage:
#   ./download_tumvi.sh          # all sequences, skipping ones already present
set -euo pipefail

DATASETS_DIR="$(cd "$(dirname "$0")" && pwd)"
TUMVI_URL="https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16"

SEQUENCES=(
  room1
  room2
  room3
  room4
  room5
  room6
)

mkdir -p "$DATASETS_DIR/tumvi"
for seq in "${SEQUENCES[@]}"; do
  name="dataset-${seq}_512_16"
  dir="$DATASETS_DIR/tumvi/$name"
  if [ -d "$dir" ]; then
    echo "skip tumvi/$name (already present)"
    continue
  fi
  echo "downloading tumvi/$name"
  wget -c -q --show-progress "$TUMVI_URL/$name.tar" -O "$DATASETS_DIR/tumvi/$name.tar"
  tar -xf "$DATASETS_DIR/tumvi/$name.tar" -C "$DATASETS_DIR/tumvi"
  rm "$DATASETS_DIR/tumvi/$name.tar"
done
