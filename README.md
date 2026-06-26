# TAEY - RGBD SLAM

Real-time RGB-D SLAM pipeline with ORB feature tracking, GTSAM iSAM2 pose-graph optimization, and TensorRT-accelerated CLIP embeddings for FAISS-based place recognition.

![TAEY SLAM Demo](media/output.gif)

## Prerequisites

- [NVIDIA Driver](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu-installations)
- [CUDA Toolkit >= 12.8](https://developer.nvidia.com/cuda-downloads)
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Getting Started

Build and enter the dev container:
```bash
docker compose build --build-arg CUDA_ARCH_BIN=$CUDA_ARCH_VERSION dev
docker compose run --remove-orphans dev

# Enter container
docker exec -it $(docker ps -lq) /bin/bash
```

All commands below run inside the container.

### Build the TensorRT Engine

Convert the CLIP ViT-B/32 model to a TensorRT engine:
```bash
python -m scripts.clip
```

This exports a `.engine` file to `models/clip/clip.engine` (input: `3×224×224`, output: 512-dim embedding).

### Build the Project

> **Note:** rerun_sdk builds a bundled Arrow → mimalloc from source, and that
> mimalloc still declares `cmake_minimum_required(VERSION <3.5)`, which CMake 4.x
> rejects. Export `CMAKE_POLICY_VERSION_MINIMUM=3.5` so the policy floor is
> inherited by those nested ExternalProject build-time `cmake` invocations.

Release:
```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -flto"
cmake --build build --config Release
```

Debug:
```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

## Usage

Run the binaries from the repo root so the relative paths in `config.yaml` resolve.

### TUM RGB-D Dataset
```bash
./build/tum [dataset_path]
```
Runs SLAM on a [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) dataset (defaults to `datasets/rgbd_dataset_freiburg2_pioneer_slam2`). Outputs estimated poses (timestamp, translation, quaternion) to `results/<scene>.txt`.

### Intel RealSense (Live)
```bash
./build/rs
```
Runs SLAM live with a connected RealSense depth camera (stream resolution/fps from `config.yaml`). Camera intrinsics are read directly from the device.

### Place Recognition Evaluation
```bash
./build/clip [dataset_path]
```
Builds a FAISS flat index from CLIP embeddings over a TUM dataset and evaluates keyframe retrieval.

## Configuration

Parameters come from two YAML files, with the dataset's calibration overlaying the repo defaults.

**`config.yaml`** (repo root) holds application defaults — model paths, the Rerun sink, and per-example run knobs:
```yaml
encoder: models/clip/clip.engine   # TensorRT engine for CLIP embeddings
rerun_save:                        # .rrd output path (headless); empty = spawn viewer
rerun_address:                     # gRPC address of a running viewer
stride: 10                         # tum: process every n-th frame
num_queries: 25                    # clip: query frames sampled from the sequence
k: 100                             # clip: nearest neighbours per query
rs_width: 640                      # rs: stream width/height/fps
rs_height: 480
rs_fps: 30
rs_margin: 0.08                    # rs: fractional crop per edge
```

**`<dataset>/calibration.yaml`** holds the per-dataset camera model, overriding any matching key in `config.yaml`:
```yaml
width: 640
height: 480
depth_scale: 5000.0   # divisor to convert raw depth to meters
fx: 517.3
fy: 516.5
cx: 318.6
cy: 255.3
distortion: [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
```

The `RERUN_SAVE` and `RERUN_ADDRESS` environment variables override `rerun_save` / `rerun_address` for one-off runs.

## Docker Targets

| Target | Base | Purpose |
|--------|------|---------|
| `dev` | `cuda:12.8-cudnn-devel` | Compilers, cmake, gdb, dev headers |
| `runtime` | `cuda:12.8-cudnn-runtime` | Shared libs only — runs pre-built binaries |

To build and run the runtime image:
```bash
docker compose build --build-arg CUDA_ARCH_BIN=$CUDA_ARCH_VERSION runtime
docker compose run --rm runtime
```
