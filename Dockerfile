FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

ENV REALSENSE_VERSION v2.57.4
ENV OPENCV_VERSION 4.12.0
ENV GTSAM_VERSION 4.3a0
ENV EIGEN_VERSION 3.4.0
ENV PCL_VERSION pcl-1.15.1

ARG CUDA_ARCH_BIN="7.5;8.9"
ARG TENSORRT_VERSION=10.8.0.43-1+cuda12.8

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        bzip2 \
        ccache \
        cmake \
        cmake-curses-gui \
        curl \
        freeglut3-dev \
        g++ \
        gdb \
        gedit \
        gfortran \
        git \
        gnupg2 \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libboost-all-dev \
        libboost-date-time-dev \
        libboost-filesystem-dev \
        libboost-program-options-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libboost-timer-dev \
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
        libtool \
        libusb-1.0-0-dev \
        libv4l-dev \
        libvtk9-dev \
        libvtk9-qt-dev \
        libwebp-dev \
        libx264-dev \
        libxvidcore-dev \
        libyaml-cpp-dev \
        locales \
        lsb-release \
        make \
        mesa-common-dev \
        mesa-utils \
        ninja-build \
        openexr \
        pkg-config \
        plocate \
        protobuf-compiler \
        python3 \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3.12-venv \
        qtbase5-dev \
        sudo \
        unzip \
        wget \
        ca-certificates \
        gpg \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
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
        tensorrt=${TENSORRT_VERSION} && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux && \
    apt-get update && \
    mkdir -p /usr/share/keyrings && \
    wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y cmake; \
    rm -rf /var/lib/apt/lists/*

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
    cmake -S pcl -B pcl/build -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build pcl/build -j"$(nproc)" && \
    cmake --install pcl/build && \
    rm -rf pcl

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
        -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
        -DGTSAM_BUILD_PYTHON=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_POLICY_DEFAULT_CMP0167=OLD && \
    cmake --build gtsam/build -j"$(nproc)" && \
    cmake --install gtsam/build && \
    rm -rf gtsam

# RMM
RUN git clone --depth=1 --branch v25.10.00 https://github.com/rapidsai/rmm.git && \
    cmake -S rmm/cpp -B rmm/build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CUDA_ARCHITECTURES="75;89" \
        -DBUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build rmm/build -j$(nproc) && \
    cmake --install rmm/build && \
    rm -rf rmm

# RAFT
RUN git clone --depth=1 --branch v25.10.00 https://github.com/rapidsai/raft.git && \
    cmake -S raft/cpp -B raft/build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CUDA_ARCHITECTURES="75;89" \
        -DBUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build raft/build -j$(nproc) && \
    cmake --install raft/build && \
    rm -rf raft

# CUVS
RUN git clone --depth=1 --branch v25.10.00 https://github.com/rapidsai/cuvs.git && \
    cmake -S cuvs/cpp -B cuvs/build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CUDA_ARCHITECTURES="75;89" \
        -DBUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build cuvs/build -j$(nproc) && \
    cmake --install cuvs/build && \
    rm -rf cuvs

# Install faiss
RUN git clone --depth=1 --branch v1.13.0 https://github.com/facebookresearch/faiss.git && \
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
        -DCMAKE_CUDA_ARCHITECTURES="75;89" && \
    cmake --build faiss/build -j$(nproc) && \
    cmake --install faiss/build --prefix /usr/local && \
    rm -rf faiss

RUN git clone --branch ${REALSENSE_VERSION} https://github.com/IntelRealSense/librealsense.git && \
    #(cd librealsense && ./scripts/setup_udev_rules.sh) && \
    cmake -S librealsense -B librealsense/build \
        -DBUILD_EXAMPLES=true \
        -DBUILD_GRAPHICAL_EXAMPLES=true && \
    cmake --build librealsense/build -j$(nproc) && \
    cmake --install librealsense/build --prefix /usr/local && \
    rm -rf librealsense

ARG USER

COPY scripts /home/${USER}/taey/
COPY requirements.txt pyproject.toml /home/${USER}/taey/

RUN python3 -m venv /opt/taey
RUN /opt/taey/bin/pip install --upgrade pip
RUN /opt/taey/bin/pip install -r /home/${USER}/taey/requirements.txt
RUN /opt/taey/bin/pip install -e /home/${USER}/taey

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

ARG TENSORRT_VERSION=10.8.0.43-1+cuda12.8

RUN apt-get remove -y 'libnvinfer*' 'tensorrt*' 'python3-libnvinfer*' 'libnvonnxparsers*' || true && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        bzip2 \
        ccache \
        cmake \
        cmake-curses-gui \
        curl \
        freeglut3-dev \
        g++ \
        gdb \
        gedit \
        gfortran \
        git \
        gnupg2 \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libboost-all-dev \
        libboost-date-time-dev \
        libboost-filesystem-dev \
        libboost-program-options-dev \
        libboost-serialization-dev \
        libboost-system-dev \
        libboost-timer-dev \
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
        libtool \
        libusb-1.0-0-dev \
        libv4l-dev \
        libvtk9-dev \
        libvtk9-qt-dev \
        libwebp-dev \
        libx264-dev \
        libxvidcore-dev \
        libyaml-cpp-dev \
        locales \
        lsb-release \
        make \
        mesa-common-dev \
        mesa-utils \
        ninja-build \
        openexr \
        pkg-config \
        plocate \
        protobuf-compiler \
        python3 \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3.12-venv \
        qtbase5-dev \
        sudo \
        unzip \
        wget \
        ca-certificates \
        gpg \
        lsb-release \
        # --- TensorRT Packages ---
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
        tensorrt=${TENSORRT_VERSION} && \
    # 3. Cleanup
    rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local /usr/local
COPY --from=build /opt/taey /opt/taey

ENV PATH=/opt/taey/bin:$PATH
ENV PATH=/usr/src/tensorrt/bin:$PATH
ENV PATH=/usr/local/cuda-12.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

ARG USER
ARG UID=1000
ARG GID=1000

RUN userdel -r ubuntu || true
RUN groupdel ubuntu || true

RUN groupadd -g ${GID} ${USER}
RUN useradd -u ${UID} -g ${GID} -m ${USER}
RUN usermod -aG video ${USER}

WORKDIR /home/${USER}/taey

RUN chown -R ${USER}:${USER} /home/${USER}/taey /opt/taey

USER ${USER}

RUN echo "source /opt/taey/bin/activate" >> ~/.bashrc

ENTRYPOINT [ "/bin/bash" ]

FROM common-runtime AS cv-dev

ARG USER
ARG UID=1000
ARG GID=1000

# Remove Ubuntu user, create ours
RUN userdel -r ubuntu || true && groupdel ubuntu || true && \
    groupadd -g ${GID} ${USER} && \
    useradd -u ${UID} -g ${GID} -m ${USER} && \
    usermod -aG video,sudo ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /home/${USER}/taey
RUN chown -R ${USER}:${USER} /home/${USER}/taey /opt/taey

USER ${USER}
RUN echo "source /opt/taey/bin/activate" >> ~/.bashrc

ENTRYPOINT [ "/bin/bash" ]

FROM runtime AS ros2

# 1. Install ROS 2 Jazzy (for Ubuntu 24.04) & Dev Tools
RUN apt-get update && apt-get install -y locales curl \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && export LANG=en_US.UTF-8 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y ros-jazzy-ros-base ros-dev-tools python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

ARG USER
ARG UID=1000
ARG GID=1000

# Remove Ubuntu user, create ours
RUN userdel -r ubuntu || true && groupdel ubuntu || true && \
    groupadd -g ${GID} ${USER} && \
    useradd -u ${UID} -g ${GID} -m ${USER} && \
    usermod -aG video,sudo ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /home/${USER}/taey
RUN chown -R ${USER}:${USER} /home/${USER}/taey /opt/taey

USER ${USER}

# Source BOTH environments (Venv + ROS)
RUN echo "source /opt/taey/bin/activate" >> ~/.bashrc
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

ENTRYPOINT [ "/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && exec bash" ]