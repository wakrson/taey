#!/bin/bash
set -e

DATASETS=(
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_xyz
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg2_xyz
    rgbd_dataset_freiburg2_desk
    rgbd_dataset_freiburg3_nostructure_texture_near_withloop
)

for DATASET in "${DATASETS[@]}"; do
    echo "=== Running $DATASET ==="

    rm -rf results/$DATASET
    rm -rf results/$DATASET.txt
    mkdir -p results/$DATASET/

    ./release_build/tum $DATASET

    evo_ape tum datasets/$DATASET/groundtruth.txt results/$DATASET.txt \
        --align --correct_scale \
        -va --plot_mode xz \
        --save_results results/$DATASET.zip \
        --save_plot results/$DATASET/$DATASET.pdf

    unzip -o results/$DATASET.zip -d results/$DATASET/
    rm -f results/$DATASET.zip
done