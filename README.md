# TAEY - RGBD SLAM

Real-time RGB-D SLAM: SIFT feature tracking with constant-velocity guided matching, GTSAM iSAM2 backend, and TensorRT-accelerated CLIP embeddings for FAISS place recognition and loop closure.

![TAEY SLAM Demo](media/out.gif)

## Prerequisites

- [NVIDIA Driver](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu-installations)
- [CUDA Toolkit >= 12.8](https://developer.nvidia.com/cuda-downloads)
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Getting Started

```bash
docker compose build --build-arg CUDA_ARCH_BIN=$CUDA_ARCH_VERSION dev
docker compose run --remove-orphans dev
```

Build the CLIP TensorRT engine (`models/clip/clip.engine`):
```bash
python -m scripts.clip
```

Build the project:
```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -flto"
cmake --build build
```

## Usage

```bash
# TUM RGB-D dataset (RERUN=0 for headless)
./build/tum [dataset_path] [results_root]

# Intel RealSense, live
./build/rs

# place-recognition evaluation
./build/clip [dataset_path]
```

Download datasets and run the benchmark:
```bash
# TUM RGB-D scenes (ORB-SLAM2 evaluation set)
datasets/download_tum_rgbd.sh

# EuRoC MAV sequences
datasets/download_euroc.sh

# TUM-VI room sequences
datasets/download_tumvi.sh

# trajectories + evo ATE into results/<timestamp>/
bash/eval_tum.sh [scene ...]
```

## Results

ATE RMSE (m), SE(3)-aligned.

### TUM RGB-D

| Sequence | ATE RMSE (m) |
|---|---|
| fr1_desk | 0.0827702073297202 |
| fr1_desk2 | 0.0981396839774766 |
| fr1_room | 0.204318755981583 |
| fr2_desk | 0.0709345795617894 |
| fr2_xyz | 0.0124287871336686 |
| fr3_long_office_household | 0.0622143742020453 |
| fr3_nostructure_texture_near_withloop | 0.0205341581873657 |

## Configuration

`config.yaml` holds application defaults; the dataset's `calibration.yaml` (camera model, depth scale) overrides matching keys.

```yaml
encoder: models/clip/clip.engine  # TensorRT engine for CLIP embeddings
max_depth: 6.0                    # discard back-projected points beyond (m)
keyframe_max_overlap: 0.5         # skip frames above this map overlap...
keyframe_min_parallax: 1.0        # skip frames below this parallax (degrees)
num_neighbors: 20                 # embedding-index neighbors for loop closure
```

## Docker Targets

| Target | Base | Purpose |
|--------|------|---------|
| `dev` | `cuda:12.8-cudnn-devel` | Compilers, cmake, gdb, dev headers |
| `runtime` | `cuda:12.8-cudnn-runtime` | Shared libs only — runs pre-built binaries |
