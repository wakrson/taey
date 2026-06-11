# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Build-time configuration (overridable with --build-arg)
# ---------------------------------------------------------------------------
ARG CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
ARG TENSORRT_VERSION=10.8.0.43-1+cuda12.8
ARG REALSENSE_VERSION=v2.57.4
ARG OPENCV_VERSION=4.12.0
ARG GTSAM_VERSION=4.3a0
ARG PCL_VERSION=pcl-1.15.1
ARG FAISS_VERSION=v1.13.0
ARG CUDA_ARCH_BIN="7.5;8.9"
ARG CUDA_ARCH_CMAKE="75;89"

# ---------------------------------------------------------------------------
# Stage 1: base — shared toolchain + dev headers used by both build and dev.
# Extracted so the package list lives in exactly one place (no drift).
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ARG TENSORRT_VERSION

# Keep apt archives so the BuildKit cache mounts below actually cache.
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Enable universe + the Kitware repo BEFORE installing cmake, so cmake is
# pulled once (from Kitware) instead of installed twice.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg2 \
        lsb-release \
        software-properties-common \
        wget && \
    add-apt-repository universe && \
    mkdir -p /usr/share/keyrings && \
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/kitware.list

# Shared compilers + dev headers (common to the build and dev stages).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        cmake \
        curl \
        freeglut3-dev \
        g++ \
        git \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libboost-all-dev \
        libcanberra-gtk-module \
        libdc1394-dev \
        libeigen3-dev \
        libflann-dev \
        libfmt-dev \
        libgflags-dev \
        libglu1-mesa \
        libgoogle-glog-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libgtk-3-dev \
        libhdf5-dev \
        libhdf5-openmpi-dev \
        libjpeg-dev \
        liblapack-dev \
        libmpich-dev \
        libnpp-dev-12-8 \
        libomp-dev \
        libopenblas-dev \
        libopenexr-dev \
        libopenjp2-7 \
        libopenmpi-dev \
        libpcap-dev \
        libpng-dev \
        libprotobuf-dev \
        libspdlog-dev \
        libswscale-dev \
        libtbb-dev \
        libtbbmalloc2 \
        libtiff-dev \
        libusb-1.0-0-dev \
        libv4l-dev \
        libvtk9-dev \
        libvtk9-qt-dev \
        libwebp-dev \
        libyaml-cpp-dev \
        locales \
        make \
        mesa-common-dev \
        mesa-utils \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3.12-venv \
        qtbase5-dev && \
    locale-gen en_US.UTF-8

# locales is installed above; actually generate and select a UTF-8 locale.
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# TensorRT dev packages (headers + libs) shared by the build and dev stages.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libnvinfer-dev=${TENSORRT_VERSION} \
        libnvinfer-dispatch-dev=${TENSORRT_VERSION} \
        libnvinfer-dispatch10=${TENSORRT_VERSION} \
        libnvinfer-headers-dev=${TENSORRT_VERSION} \
        libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
        libnvinfer-lean-dev=${TENSORRT_VERSION} \
        libnvinfer-lean10=${TENSORRT_VERSION} \
        libnvinfer-plugin-dev=${TENSORRT_VERSION} \
        libnvinfer-plugin10=${TENSORRT_VERSION} \
        libnvinfer-vc-plugin-dev=${TENSORRT_VERSION} \
        libnvinfer-vc-plugin10=${TENSORRT_VERSION} \
        libnvinfer10=${TENSORRT_VERSION} \
        libnvonnxparsers-dev=${TENSORRT_VERSION} \
        libnvonnxparsers10=${TENSORRT_VERSION} \
        tensorrt-dev=${TENSORRT_VERSION} \
        tensorrt-libs=${TENSORRT_VERSION}

# ---------------------------------------------------------------------------
# Stage 2: build — compile third-party deps from source, build the venv.
# ---------------------------------------------------------------------------
FROM base AS build

ARG TENSORRT_VERSION
ARG CUDA_ARCH_BIN
ARG CUDA_ARCH_CMAKE
ARG REALSENSE_VERSION
ARG OPENCV_VERSION
ARG GTSAM_VERSION
ARG PCL_VERSION
ARG FAISS_VERSION

# ccache shared across builds via a BuildKit cache mount (see RUNs below).
ENV CCACHE_DIR=/ccache

# Build-only tooling + the extra TensorRT packages the runtime/dev don't need.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        bzip2 \
        gfortran \
        libtool \
        libx264-dev \
        libxvidcore-dev \
        openexr \
        protobuf-compiler \
        unzip \
        libnvinfer-bin=${TENSORRT_VERSION} \
        libnvinfer-samples=${TENSORRT_VERSION} \
        python3-libnvinfer-dev=${TENSORRT_VERSION} \
        python3-libnvinfer-dispatch=${TENSORRT_VERSION} \
        python3-libnvinfer-lean=${TENSORRT_VERSION} \
        python3-libnvinfer=${TENSORRT_VERSION} \
        tensorrt=${TENSORRT_VERSION}

RUN --mount=type=cache,target=/ccache \
    git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git && \
    git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git && \
    cmake -S opencv -B opencv/build \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_CUBLAS=ON \
        -DWITH_TBB=ON \
        -DOPENCV_DNN_CUDA=ON \
        -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
        -DBUILD_opencv_hdf=OFF \
        -DBUILD_TESTS=OFF \
        -DOPENCV_ENABLE_NONFREE=ON && \
    cmake --build opencv/build -j"$(nproc)" && \
    cmake --install opencv/build && \
    rm -rf opencv opencv_contrib

RUN --mount=type=cache,target=/ccache \
    git clone --branch ${PCL_VERSION} --depth 1 https://github.com/PointCloudLibrary/pcl.git && \
    cmake -S pcl -B pcl/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache && \
    cmake --build pcl/build -j"$(nproc)" && \
    cmake --install pcl/build && \
    rm -rf pcl

RUN --mount=type=cache,target=/ccache \
    git clone --branch ${GTSAM_VERSION} --depth 1 https://github.com/borglab/gtsam.git && \
    cmake -S gtsam -B gtsam/build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DGTSAM_USE_BOOST_FEATURES=OFF \
        -DGTSAM_ENABLE_BOOST_SERIALIZATION=ON \
        -DGTSAM_BUILD_TESTS=OFF \
        -DGTSAM_BUILD_EXAMPLES=OFF \
        -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
        -DGTSAM_WITH_TBB=ON \
        -DGTSAM_USE_SYSTEM_EIGEN=ON \
        -DGTSAM_BUILD_SHARED_LIBS=ON \
        -DGTSAM_BUILD_UNSTABLE=OFF \
        -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
        -DGTSAM_BUILD_PYTHON=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_POLICY_DEFAULT_CMP0167=OLD \
        -DCMAKE_CXX_FLAGS="-Wno-error=array-bounds -Wno-error=stringop-overflow -Wno-error=overloaded-virtual" && \
    cmake --build gtsam/build -j"$(nproc)" && \
    cmake --install gtsam/build && \
    rm -rf gtsam

RUN --mount=type=cache,target=/ccache \
    git clone --depth 1 --branch ${FAISS_VERSION} https://github.com/facebookresearch/faiss.git && \
    cmake -S faiss -B faiss/build \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_CUVS=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_C_API=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_OPT_LEVEL=avx2 \
        -DFAISS_USE_LTO=ON \
        -DCUDAToolkit_ROOT=/usr/local/cuda \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_CMAKE}" && \
    cmake --build faiss/build -j"$(nproc)" && \
    cmake --install faiss/build --prefix /usr/local && \
    rm -rf faiss

# librealsense: shallow clone, no examples — we only need the SDK library.
RUN --mount=type=cache,target=/ccache \
    git clone --branch ${REALSENSE_VERSION} --depth 1 https://github.com/IntelRealSense/librealsense.git && \
    cmake -S librealsense -B librealsense/build \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=false \
        -DBUILD_GRAPHICAL_EXAMPLES=false && \
    cmake --build librealsense/build -j"$(nproc)" && \
    cmake --install librealsense/build --prefix /usr/local && \
    rm -rf librealsense

ARG USER

COPY scripts /home/${USER}/taey/
COPY pyproject.toml README.md /home/${USER}/taey/

# Index URLs that used to live in requirements.txt are passed here, since
# pyproject.toml has no field for them. --extra-index-url preserves the same
# resolution behavior the requirements.txt had.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/taey --system-site-packages && \
    /opt/taey/bin/pip install --upgrade pip && \
    /opt/taey/bin/pip install \
        --extra-index-url https://pypi.nvidia.com \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e /home/${USER}/taey

# Pruned /usr/local for the runtime image: shared libs (stripped) + binaries +
# data, but no headers, static archives, or cmake/pkgconfig metadata.
RUN mkdir -p /runtime/usr/local && \
    cp -a /usr/local/lib /runtime/usr/local/ && \
    cp -a /usr/local/bin /runtime/usr/local/ 2>/dev/null || true && \
    cp -a /usr/local/share /runtime/usr/local/ 2>/dev/null || true && \
    rm -rf /runtime/usr/local/lib/cmake /runtime/usr/local/lib/pkgconfig && \
    find /runtime/usr/local/lib -name '*.a' -delete && \
    find /runtime/usr/local/lib -type f \( -name '*.so' -o -name '*.so.*' \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 3: dev — compilers, dev headers, debug tools for active development.
# ---------------------------------------------------------------------------
FROM base AS dev

# Dev-only extras on top of the shared base toolchain.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gdb \
        sudo

# Claude Code CLI — dev convenience only (unpinned upstream installer; kept out
# of the runtime image deliberately).
RUN curl -fsSL https://claude.ai/install.sh | bash

COPY --from=build /usr/local /usr/local
COPY --from=build /opt/taey /opt/taey

ENV PATH=/opt/taey/bin:/usr/src/tensorrt/bin:/usr/local/cuda-12.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

ARG USER
ARG UID=1000
ARG GID=1000

RUN userdel -r ubuntu || true && \
    groupdel ubuntu || true && \
    groupadd -g ${GID} ${USER} && \
    useradd -u ${UID} -g ${GID} -m ${USER} && \
    usermod -aG video ${USER}

WORKDIR /home/${USER}/dev/taey

RUN chown -R ${USER}:${USER} /home/${USER}/dev/taey /opt/taey

USER ${USER}

RUN echo "source /opt/taey/bin/activate" >> ~/.bashrc

ENTRYPOINT [ "/bin/bash" ]

# ---------------------------------------------------------------------------
# Stage 4: runtime — minimal image for running pre-built binaries.
# ---------------------------------------------------------------------------
FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ARG TENSORRT_VERSION

RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Runtime shared libraries only (no compilers, no -dev headers).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libglut3.12 \
        libatlas3-base \
        libavcodec60 \
        libavformat60 \
        libboost-filesystem1.83.0 \
        libboost-serialization1.83.0 \
        libboost-system1.83.0 \
        libcanberra-gtk-module \
        libdc1394-25 \
        libflann1.9 \
        libfmt9 \
        libgflags2.2 \
        libglu1-mesa \
        libgoogle-glog0v6t64 \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer1.0-0 \
        libgtk-3-0t64 \
        libhdf5-openmpi-103-1t64 \
        libjpeg-turbo8 \
        liblapack3 \
        libomp5 \
        libopenblas0 \
        libopenexr-3-1-30 \
        libopenjp2-7 \
        libopenmpi3t64 \
        libpng16-16t64 \
        libprotobuf32t64 \
        libspdlog1.12 \
        libswscale7 \
        libtbb12 \
        libtbbmalloc2 \
        libtiff6 \
        libusb-1.0-0 \
        libv4l-0 \
        libvtk9.1-qt \
        libwebp7 \
        libyaml-cpp0.8 \
        locales \
        mesa-utils \
        python3 \
        python3-numpy \
        qt5-gtk-platformtheme && \
    locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# TensorRT runtime libs only.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libnvinfer-dispatch10=${TENSORRT_VERSION} \
        libnvinfer-lean10=${TENSORRT_VERSION} \
        libnvinfer-plugin10=${TENSORRT_VERSION} \
        libnvinfer-vc-plugin10=${TENSORRT_VERSION} \
        libnvinfer10=${TENSORRT_VERSION} \
        libnvonnxparsers10=${TENSORRT_VERSION} \
        tensorrt-libs=${TENSORRT_VERSION}

# Pruned prefix from the build stage (shared libs only, no headers/static).
COPY --from=build /runtime/usr/local /usr/local
COPY --from=build /opt/taey /opt/taey

ENV PATH=/opt/taey/bin:/usr/local/cuda-12.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

ARG USER
ARG UID=1000
ARG GID=1000

RUN userdel -r ubuntu || true && \
    groupdel ubuntu || true && \
    groupadd -g ${GID} ${USER} && \
    useradd -u ${UID} -g ${GID} -m ${USER} && \
    usermod -aG video ${USER}

WORKDIR /home/${USER}/dev/taey

RUN chown -R ${USER}:${USER} /home/${USER}/dev/taey /opt/taey

USER ${USER}

RUN echo "source /opt/taey/bin/activate" >> ~/.bashrc

ENTRYPOINT [ "/bin/bash" ]
