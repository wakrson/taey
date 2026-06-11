#pragma once

#include <filesystem>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "engine.h"

class CLIP {
public:
  CLIP(const std::string &path);
  Eigen::VectorXf operator()(const cv::Mat &);
  Eigen::VectorXf encode(const cv::Mat &);
  Eigen::VectorXf encode(const cv::cuda::GpuMat &);
  static double cosineDistance(const Eigen::VectorXf &, const Eigen::VectorXf &);

private:
  bool load(const std::string &);
  std::vector<std::vector<cv::cuda::GpuMat>>
  preprocess(const cv::cuda::GpuMat &);

  static constexpr bool NORMALIZE = true;
  static constexpr std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
  static constexpr std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
  std::unique_ptr<Engine<float>> engine_;

  // Reusable GPU buffers
  cv::cuda::GpuMat rgb_mat_;
  cv::cuda::GpuMat resized_;
};
