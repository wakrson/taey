# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

cmake_minimum_required(VERSION 3.22)

# This is here so we can use jthread from C++ 20
set(CMAKE_CXX_STANDARD 20)

project(taey_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
set(THIS_PACKAGE_INCLUDE_DEPENDS
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  std_msgs
)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

# Add the hardware interface
add_library(
  taey_node
  SHARED
  src/taey_node.cpp
)
target_compile_features(taey_node PUBLIC cxx_std_20)
target_include_directories(taey_node PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(
  taey_node PUBLIC
  ${THIS_PACKAGE_INCLUDE_DEPENDS}
)

# Add example nodes and install them
add_executable(taey src/taey_node.cpp)
target_link_libraries(taey taey_node)

install(TARGETS noarm_squat wiggle_arm joint_command_passthrough
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS taey_node
  EXPORT export_taey_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(export_taey_node HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

ament_package()