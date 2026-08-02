#pragma once

#include <stdint.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

class MapPoint;
class FramePoint;
class Camera;

class KeyFrame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  KeyFrame(const std::size_t &, const double &, cv::Mat, cv::Mat, const YAML::Node &);
  ~KeyFrame();

  std::size_t numMapPoints() const;
  std::vector<std::shared_ptr<MapPoint>> mapPoints();
  cv::Mat image() const;
  cv::Mat depth() const;
  std::shared_ptr<Camera> camera() const;
  void setPose(const Eigen::Transform<double, 3, Eigen::Isometry> &);
  Eigen::Transform<double, 3, Eigen::Isometry> pose() const;
  std::vector<std::shared_ptr<FramePoint>> framePoints();
  std::size_t id() const;
  double timestamp();
  const Eigen::VectorXf &imageEmbedding() const;
  void imageEmbedding(const Eigen::VectorXf &embedding);
  bool estimatePose(Eigen::Transform<double, 3, Eigen::Isometry> &);
  std::size_t numFramePoints() const;
  void objectPoints(Eigen::MatrixXd&) const;

private:
  cv::Mat depth_, image_;
  std::size_t id_;
  double timestamp_;
  std::shared_ptr<Camera> camera_;
  std::vector<std::shared_ptr<FramePoint>> frame_points_;
  Eigen::VectorXf image_embedding_;
  mutable std::mutex pose_mtx_;
};
