# TAEY - RGBD SLAM Implementation

Real-time RGB-D SLAM pipeline leveraging TensorRT-accelerated ViT-B/32 embeddings for FAISS-based keyframe retrieval and GTSAM iSAM2 for incremental pose-graph optimization

View Project →
![TAEY SLAM Demo](media/output.gif)


## Prerequisites
```bash
# NVIDIA Driver
https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu-installations

# CUDA Toolkit (CUDA >= 12.8)
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#id6
https://developer.nvidia.com/cuda-downloads

# Docker Engine
https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

# NVIDIA Container Toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```

## Getting Started
```bash
# Get CUDA arch binary version: https://developer.nvidia.com/cuda-gpus
# Build docker container
docker compose build --build-arg CUDA_ARCH_BIN=8.9

# Run container
docker compose run --remove-orphans taey

# Compile ViT image encoder
python3 models/encoder.py

# Build and install source code
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -flto"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

cmake --build build --config Release -j$(nproc)

# Run tum
./build/tum

# Run realsense
./build/rs
```