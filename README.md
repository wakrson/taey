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

### TUM RGB-D Dataset
```bash
./build/tum
```
Runs SLAM on a [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) dataset. Outputs estimated poses (timestamp, translation, quaternion) to a file.

### Intel RealSense (Live)
```bash
./build/rs
```
Runs SLAM live with a connected RealSense depth camera (640×480 @ 30 fps). Camera intrinsics are read directly from the device.

### Place Recognition Evaluation
```bash
./build/clip
```
Builds a FAISS flat index from CLIP embeddings over a TUM dataset and evaluates keyframe retrieval.

## Configuration

Each dataset directory requires a `calibration.yaml`:
```yaml
width: 640
height: 480
depth_scale: 5000.0   # divisor to convert raw depth to meters
fx: 517.3
fy: 516.5
cx: 318.6
cy: 255.3
distoration: [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
```

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
