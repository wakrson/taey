cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

if (POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# Force absolute paths in compilation output
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_USE_RELATIVE_PATHS ON)

# Optional: verbose output for make

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.8)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

add_library(tensorrt_cpp_api SHARED src/engine.cpp)

# Correct message to show TensorRT include dirs
message(STATUS "TensorRT include dirs: ${TensorRT_INCLUDE_DIRS}")
message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")

target_include_directories(tensorrt_cpp_api 
    PUBLIC 
        ${OpenCV_INCLUDE_DIRS} 
        ${CUDA_INCLUDE_DIRS} 
        ${TensorRT_INCLUDE_DIRS} 
        include 
        include/interfaces
)

target_link_libraries(tensorrt_cpp_api 
    PUBLIC 
        ${OpenCV_LIBS} 
        ${CUDA_LIBRARIES} 
        ${CMAKE_THREAD_LIBS_INIT} 
        ${TensorRT_LIBRARIES} 
        fmt::fmt
)

add_executable(run_inference_benchmark src/main.cpp)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api fmt::fmt)
