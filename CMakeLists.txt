cmake_minimum_required(VERSION 3.14)
project(taey VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
find_package(PCL REQUIRED COMPONENTS common io visualization filters)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(FAISS REQUIRED)

# --- core library ---
set(TAEY_HEADERS
  include/taey/ImageEncoder.h
  include/taey/Camera.h
  include/taey/FramePoint.h
  include/taey/KeyFrame.h
  include/taey/Map.h
  include/taey/MapPoint.h
  include/taey/Optimizer.h
  include/taey/TAEY.h
  include/taey/utils.h
  include/taey/types.h
  include/taey/TUM.h
  include/taey/Visualizer.h
)

set(TAEY_SRCS
  src/ImageEncoder.cpp
  src/Camera.cpp
  src/FramePoint.cpp
  src/KeyFrame.cpp
  src/Map.cpp
  src/MapPoint.cpp
  src/Optimizer.cpp
  src/TAEY.cpp
  src/utils.cpp
  src/TUM.cpp
  src/Visualizer.cpp
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
    Threads::Threads
    tensorrt_cpp_api
    yaml-cpp
    Qt5::Widgets
    Qt5::OpenGL
    faiss
)

if(PCL_DEFINITIONS)
  target_compile_definitions(taey PRIVATE ${PCL_DEFINITIONS})
endif()

target_compile_options(taey PRIVATE -Wno-deprecated-declarations)

add_executable(tum
  examples/tum.cpp
)

# Link and include for executable
target_link_libraries(tum PRIVATE taey yaml-cpp tensorrt_cpp_api faiss)

target_include_directories(tum PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/libs/tensorrt-cpp-api/include
)

# --- Install ---
install(TARGETS taey tum
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
