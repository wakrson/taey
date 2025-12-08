#pragma once

#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "engine.h"

class KeyFrame;

class ImageEncoder {
public:
  ImageEncoder();
  ImageEncoder(const std::string &p);
  bool load(const std::string &p);
  Eigen::VectorXf operator()(const cv::Mat &);
  double cosineDistance(const Eigen::VectorXf &, const Eigen::VectorXf &);
  std::vector<std::vector<cv::cuda::GpuMat>>
  preprocess(const cv::cuda::GpuMat &);

private:
  const bool NORMALIZE = true;
  const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
  const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
  std::unique_ptr<Engine<float>> trt_engine_;
};