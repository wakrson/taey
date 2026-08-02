#!/bin/bash
# Run the tum binary on TUM RGB-D scenes, evaluate ATE with evo.
# Each invocation writes into a fresh results/<timestamp>/ run folder:
# per scene <scene>/{<scene>.txt, ape.pdf, evo result files}, plus
# summary.csv and summary.pdf across scenes.
set -euo pipefail

cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
evo_config set plot_backend Agg >/dev/null   # plot to file without a display

run="results/$(date +%s)"

SCENES=("${@:-rgbd_dataset_freiburg1_desk \
  rgbd_dataset_freiburg1_desk2 \
  rgbd_dataset_freiburg1_room \
  rgbd_dataset_freiburg2_desk \
  rgbd_dataset_freiburg2_xyz \
  rgbd_dataset_freiburg3_long_office_household \
  rgbd_dataset_freiburg3_nostructure_texture_near_withloop}")

zips=()
for scene in ${SCENES[@]}; do
  echo "== $scene"
  RERUN=0 ./build/tum "datasets/$scene" "$run"
  evo_ape tum "datasets/$scene/groundtruth.txt" "$run/$scene/$scene.txt" -a \
    --save_results "$tmp/$scene.zip" --save_plot "$run/$scene/ape.pdf" | tail -8
  # Flatten the evo results next to the pdf; keep our trajectory, not evo's copy.
  unzip -oq "$tmp/$scene.zip" -x "$scene.txt" -d "$run/$scene"
  zips+=("$tmp/$scene.zip")
done

evo_res "${zips[@]}" --save_table "$run/summary.csv" --save_plot "$run/summary.pdf"