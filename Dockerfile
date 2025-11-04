FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV OPENCV_VERSION 4.12.0
ENV GTSAM_VERSION 4.3a0
ENV EIGEN_VERSION 3.4.0
ENV PCL_VERSION pcl-1.15.1

ARG TENSORRT_VERSION=10.8.0.43-1+cuda12.8
ARG CUDA_ARCH_BIN=7.5

RUN apt update && \
    apt install -y \
        build-essential \
        wget \
        bzip2 \
        git \
        cmake \
        g++ \
        locales \
        software-properties-common \
        sudo \
        curl \
        gnupg2 \
        lsb-release \
        gedit \
        unzip \
        pkg-config \
        libgtk-3-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        gfortran \
        openexr \
        libatlas-base-dev \
        libtbb2 \
        libtbb-dev \
        libdc1394-dev \
        libopenexr-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libopenjp2-7 \
        autoconf \
        automake \
        libtool \
        make \
        mlocate \
        libusb-1.0-0-dev \
        cmake-curses-gui \
        mesa-common-dev \
        mesa-utils \
        freeglut3-dev \
        ninja-build \
        libopenmpi-dev \
        libmpich-dev \
        libopenmpi-dev \
        libhdf5-openmpi-dev \
        libeigen3-dev \
        libtbb-dev \
        libflann-dev \
        software-properties-common && \
    add-apt-repository universe

RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-timer-dev \
    libboost-system-dev \
    libboost-serialization-dev \
    libboost-date-time-dev \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget gpg software-properties-common ca-certificates; \
    mkdir -p /usr/share/keyrings; \
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg; \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
        > /etc/apt/sources.list.d/kitware.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends cmake; \
    cmake --version; \
    rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install -y \
        python3 \
        python3-dev \
        python3-pip

RUN apt-get update && apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libpng-dev libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    libtbb2 \
    libtbb-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    python3-numpy \
    libnpp-dev-12-8 \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install -y \
        qtbase5-dev \
        libvtk9-dev \
        libvtk9-qt-dev

RUN echo "CUDA_PATH=$CUDA_PATH" && \
    echo "CUDACXX=$CUDACXX" && \
    echo "PATH=$PATH" && \
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LIBRARY_PATH
ENV CMAKE_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$CMAKE_LIBRARY_PATH

RUN git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git && \
	git clone --branch ${OPENCV_VERSION} --depth 1  https://github.com/opencv/opencv_contrib.git && \
    cmake -S opencv -B opencv/build \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_TBB=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
        -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
        -D BUILD_opencv_hdf=OFF \
        -D BUILD_TESTS=OFF \
        -D OPENCV_ENABLE_NONFREE=ON .. && \
	cmake --build opencv/build -j"$(nproc)" && \
	cmake --install opencv/build && \
    rm -rf opencv && \
    rm -rf opencv_contrib

RUN git clone --branch ${PCL_VERSION} --depth 1 https://github.com/PointCloudLibrary/pcl.git && \
    cmake -S pcl -B pcl/build \
        -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build pcl/build -j"$(nproc)" && \
    cmake --install pcl/build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libnvinfer-bin=${TENSORRT_VERSION} \
    libnvinfer-dev=${TENSORRT_VERSION} \
    libnvinfer-dispatch-dev=${TENSORRT_VERSION} \
    libnvinfer-dispatch10=${TENSORRT_VERSION} \
    libnvinfer-headers-dev=${TENSORRT_VERSION} \
    libnvinfer-headers-plugin-dev=${TENSORRT_VERSION} \
    libnvinfer-lean-dev=${TENSORRT_VERSION} \
    libnvinfer-lean10=${TENSORRT_VERSION} \
    libnvinfer-plugin-dev=${TENSORRT_VERSION} \
    libnvinfer-plugin10=${TENSORRT_VERSION} \
    libnvinfer-samples=${TENSORRT_VERSION} \
    libnvinfer-vc-plugin-dev=${TENSORRT_VERSION} \
    libnvinfer-vc-plugin10=${TENSORRT_VERSION} \
    libnvinfer10=${TENSORRT_VERSION} \
    libnvonnxparsers-dev=${TENSORRT_VERSION} \
    libnvonnxparsers10=${TENSORRT_VERSION} \
    python3-libnvinfer-dev=${TENSORRT_VERSION} \
    python3-libnvinfer-dispatch=${TENSORRT_VERSION} \
    python3-libnvinfer-lean=${TENSORRT_VERSION} \
    python3-libnvinfer=${TENSORRT_VERSION} \
    tensorrt-dev=${TENSORRT_VERSION} \
    tensorrt-libs=${TENSORRT_VERSION} \
    tensorrt=${TENSORRT_VERSION} \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --branch ${GTSAM_VERSION} --depth 1 https://github.com/borglab/gtsam.git && \
    cmake -S gtsam -B gtsam/build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DGTSAM_USE_BOOST_FEATURES=ON \
        -DGTSAM_ENABLE_BOOST_SERIALIZATION=ON \
        -DGTSAM_BUILD_TESTS=OFF \
        -DGTSAM_BUILD_EXAMPLES=OFF \
        -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
        -DGTSAM_WITH_TBB=OFF \
        -DGTSAM_USE_SYSTEM_EIGEN=ON \
        -DGTSAM_BUILD_SHARED_LIBS=ON \
        -DGTSAM_BUILD_UNSTABLE=OFF \
        -DGTSAM_BUILD_WITH_MARCH_NATIVE=OF \
        -DGTSAM_BUILD_PYTHON=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_POLICY_DEFAULT_CMP0167=OLD && \
    cmake --build gtsam/build -j"$(nproc)" && \
    cmake --install gtsam/build && \
    rm -rf gtsam

RUN apt update && \
    apt install -y \
        ccache \
        libpcap-dev \
        libspdlog-dev \
        libfmt-dev \
        libyaml-cpp-dev \
        gdb \
        libglu1-mesa \
        libgtk-3-dev \
        libcanberra-gtk-module \
        libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# CUVS
RUN git clone --depth=1 https://github.com/rapidsai/cuvs.git && \
    cd cuvs && \
    INSTALL_PREFIX=/usr/local ./build.sh libcuvs && \
    cd .. && \
    rm -rf cuvs

# RMM
RUN git clone --depth=1 https://github.com/rapidsai/rmm.git && \
    cd rmm && \
    INSTALL_PREFIX=/usr/local ./build.sh librmm && \
    cd .. && \
    rm -rf rmm

# RAFT
RUN git clone --depth=1 https://github.com/rapidsai/raft.git && \
    cd raft && \
    INSTALL_PREFIX=/usr/local ./build.sh libraft && \
    cd .. && \
    rm -rf raft

# Install faiss
RUN git clone --depth=1 https://github.com/facebookresearch/faiss.git && \
    cmake -S faiss -B faiss/build \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_CUVS=ON \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_ENABLE_C_API=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_OPT_LEVEL=avx2 \
        -DFAISS_USE_LTO=ON \
        -DCUDAToolkit_ROOT=/usr/local/cuda \
        -DCMAKE_CUDA_ARCHITECTURES="75;86" && \
    cmake --build faiss/build -j$(nproc) && \
    cmake --install faiss/build --prefix /usr/local && \
    rm -rf faiss

RUN apt update && \
    apt install -y clang-format clang-tidy

ARG USERNAME
RUN useradd -m ${USERNAME}
RUN usermod -aG video ${USERNAME}
USER ${USERNAME}

WORKDIR /home/${USERNAME}/dev/taey

ENV PATH=/usr/src/tensorrt/bin:$PATH

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "/bin/bash" ]