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
  /**
   * @brief Default constructor
   */
  ImageEncoder();
  /**
   * @brief Constructor with path to model
   * @param p: Path to model
   */
  ImageEncoder(const std::string &p);
  /**
   * @brief Load model from path
   * @param p: Path to model
   * @return Success flag
   */
  bool load(const std::string &p);
  /**
   * @brief Visually compare the distance between the keyframes
   * @param kf1: Query keyframe
   * @param kf2: Train keyframe
   * @return Cosine distance between both scenes
   */
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