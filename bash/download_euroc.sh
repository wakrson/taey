#!/bin/bash
#
# Download the 11 EuRoC MAV sequences from the ORB-SLAM3 evaluation (Campos
# et al., T-RO 2021) into datasets/euroc/.
#
# Usage:
#   ./download_euroc.sh          # all sequences, skipping ones already present
set -euo pipefail

DATASETS_DIR="$(cd "$(dirname "$0")" && pwd)"
EUROC_URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

SEQUENCES=(
  machine_hall/MH_01_easy
  machine_hall/MH_02_easy
  machine_hall/MH_03_medium
  machine_hall/MH_04_difficult
  machine_hall/MH_05_difficult
  vicon_room1/V1_01_easy
  vicon_room1/V1_02_medium
  vicon_room1/V1_03_difficult
  vicon_room2/V2_01_easy
  vicon_room2/V2_02_medium
  vicon_room2/V2_03_difficult
)

for seq in "${SEQUENCES[@]}"; do
  name="${seq##*/}"
  dir="$DATASETS_DIR/euroc/$name"
  if [ -d "$dir" ]; then
    echo "skip euroc/$name (already present)"
    continue
  fi
  echo "downloading euroc/$name"
  mkdir -p "$dir"
  wget -c -q --show-progress "$EUROC_URL/$seq/$name.zip" -O "$dir/$name.zip"
  unzip -q "$dir/$name.zip" -d "$dir"
  rm "$dir/$name.zip"
done
