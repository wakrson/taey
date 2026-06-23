cmake_minimum_required(VERSION 3.14)
project(taey VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# taey is built as a SHARED library that links the static rerun_sdk archive and
# its bundled deps. Those objects must be position-independent or the shared
# link fails with "relocation R_X86_64_PC32 ... can not be used when making a
# shared object" (the Release build only avoids this incidentally via LTO).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Debug build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Build flags per configuration
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# Base warnings and policies
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wunused
    -Wunused-function
    -Wunused-variable
    -Wunused-parameter
    -Wunused-but-set-variable
    -Wunused-but-set-parameter
    -Wunreachable-code
    -Wmissing-declarations
)

option(TAEY_BUILD_TESTS "Build tests" OFF)

# --- Handle CMake policies for compatibility ---
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)   # Allow FindBoost inside PCL
endif()
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)   # Ignore legacy EIGEN_ROOT
endif()
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)   # Ignore EIGEN_ROOT legacy var
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)   # Allow FindBoost in PCL
endif()

# --- Ignore Conda completely, even if it's active ---
if(DEFINED ENV{CONDA_PREFIX})
  message(STATUS "Conda detected at $ENV{CONDA_PREFIX} - forcing system libraries only")

  # Ignore Conda libs/includes
  list(APPEND CMAKE_IGNORE_PATH
       "$ENV{CONDA_PREFIX}/include"
       "$ENV{CONDA_PREFIX}/lib")

  # Only search system paths
  set(CMAKE_SYSTEM_PREFIX_PATH "/usr;/usr/local")
  set(CMAKE_PREFIX_PATH "/usr;/usr/local")

  # Clean pkg-config (force system .pc files)
  set(ENV{PKG_CONFIG_PATH} "/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig")

  # Runtime: only system paths
  set(CMAKE_SKIP_RPATH FALSE)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu;/usr/local/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

# --- deps ---
find_package(Eigen3 3.4.0 REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS system filesystem REQUIRED)
find_package(GTSAM 4.3 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

# --- Eigen alignment ABI: keep us consistent with the installed GTSAM ---
# Every PCL imported target carries INTERFACE_COMPILE_OPTIONS of
#   $<$<COMPILE_LANGUAGE:CXX>:-msse4.2;-mfpmath=sse;-march=native;-mavx2>
# which propagates to anything linking PCL (our `taey` lib + the executables).
# `-march=native -mavx2` raise Eigen's alignment to 32 bytes (AVX), but the system
# GTSAM is built with GTSAM_BUILD_WITH_MARCH_NATIVE=OFF — i.e. 16-byte Eigen.
# GTSAM factor templates (e.g. PriorFactor<Pose3>) are header code instantiated
# *into our binary*: with AVX they allocate the error/Jacobian vectors via Eigen's
# handmade_aligned_malloc (offset pointer), while precompiled libgtsam frees them
# with plain free() during ISAM2 linearize — a heap corruption / "bad free 32 bytes
# inside region" crash. Strip the AVX/native flags from the PCL targets so our Eigen
# alignment matches GTSAM (16-byte). -msse4.2 is kept; it leaves Eigen at 16-byte.
foreach(_pcl_tgt ${PCL_LIBRARIES})
  if(TARGET ${_pcl_tgt})
    get_target_property(_opts ${_pcl_tgt} INTERFACE_COMPILE_OPTIONS)
    if(_opts)
      string(REPLACE "-march=native" "" _opts "${_opts}")
      string(REPLACE "-mavx2" "" _opts "${_opts}")
      string(REPLACE ";;" ";" _opts "${_opts}")
      set_target_properties(${_pcl_tgt} PROPERTIES INTERFACE_COMPILE_OPTIONS "${_opts}")
    endif()
  endif()
endforeach()

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(FAISS REQUIRED)
find_package(realsense2 REQUIRED)
find_library(PCAP_LIBRARY NAMES pcap)
if(PCAP_LIBRARY)
  message(STATUS "Found pcap: ${PCAP_LIBRARY}")
else()
  message(WARNING "libpcap not found — libpcl_io.so may have unresolved symbols")
endif()

message(STATUS "Eigen3 found at: ${EIGEN3_INCLUDE_DIR} (Version: ${EIGEN3_VERSION})")

# --- rerun C++ SDK (visualization) ---
# Prefer a prebuilt SDK baked into the container image (see Dockerfile); fall
# back to fetching the matching release at configure time when building outside
# the image. Both paths expose the `rerun_sdk` target.
set(RERUN_SDK_DIR "/opt/rerun_cpp_sdk" CACHE PATH "Pre-extracted rerun C++ SDK")
if(EXISTS "${RERUN_SDK_DIR}/CMakeLists.txt")
  message(STATUS "Using pre-extracted rerun C++ SDK at ${RERUN_SDK_DIR}")
  add_subdirectory(${RERUN_SDK_DIR} ${CMAKE_BINARY_DIR}/rerun_sdk)
else()
  message(STATUS "rerun C++ SDK not found locally - fetching release 0.28.2")
  include(FetchContent)
  FetchContent_Declare(rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/download/0.28.2/rerun_cpp_sdk.zip)
  FetchContent_MakeAvailable(rerun_sdk)
endif()

# --- core library ---
set(TAEY_HEADERS
  include/taey/CLIP.h
  include/taey/Camera.h
  include/taey/FramePoint.h
  include/taey/KeyFrame.h
  include/taey/Map.h
  include/taey/MapPoint.h
  include/taey/Optimizer.h
  include/taey/TAEY.h
  include/taey/TUM.h
)

set(TAEY_SRCS
  src/CLIP.cpp
  src/Camera.cpp
  src/FramePoint.cpp
  src/KeyFrame.cpp
  src/Map.cpp
  src/MapPoint.cpp
  src/Optimizer.cpp
  src/TAEY.cpp
  src/TUM.cpp
)

# --- Add TensorRT C++ API subdirectory ---
add_subdirectory(libs/tensorrt-cpp-api SYSTEM)

# --- Build taey library ---
add_library(taey SHARED ${TAEY_SRCS} ${TAEY_HEADERS})

# Suppress warnings from its source files too
get_property(TENSORRT_TARGETS DIRECTORY libs/tensorrt-cpp-api PROPERTY BUILDSYSTEM_TARGETS)
foreach(t ${TENSORRT_TARGETS})
    target_compile_options(${t} PRIVATE -w)
endforeach()

# Include directories
target_include_directories(taey PUBLIC
    ${PROJECT_SOURCE_DIR}/include/taey
    SYSTEM ${PROJECT_SOURCE_DIR}/libs/tensorrt-cpp-api/include
    SYSTEM ${PROJECT_SOURCE_DIR}/libs/tensorrt-cpp-api/src
)

target_link_libraries(taey
  PUBLIC
    Eigen3::Eigen
    gtsam
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    $<$<BOOL:${PCAP_LIBRARY}>:${PCAP_LIBRARY}>
    Threads::Threads
    tensorrt_cpp_api
    yaml-cpp
    Qt5::Widgets
    rerun_sdk
    faiss
)

if(PCL_DEFINITIONS)
  target_compile_definitions(taey PRIVATE ${PCL_DEFINITIONS})
endif()

target_compile_options(taey PRIVATE -Wno-deprecated-declarations)

add_executable(tum examples/tum.cpp)
add_executable(rs examples/rs.cpp)
add_executable(clip examples/clip.cpp)

# Link and include for executable
target_link_libraries(tum PRIVATE taey yaml-cpp tensorrt_cpp_api faiss)
target_link_libraries(rs PRIVATE taey yaml-cpp tensorrt_cpp_api faiss ${realsense2_LIBRARY})
target_link_libraries(clip PRIVATE taey yaml-cpp tensorrt_cpp_api faiss)

target_include_directories(tum PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/libs/tensorrt-cpp-api/include
)

target_include_directories(rs PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/libs/tensorrt-cpp-api/include
)

target_include_directories(clip PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/libs/tensorrt-cpp-api/include
)

# --- Install ---
install(TARGETS taey tum rs clip
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
