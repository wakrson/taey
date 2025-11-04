#pragma once

#include <iostream>
#include <map>
#include <stdint.h>
#include <string>
#include <unordered_set>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/common/projection_matrix.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/point_types.hpp>

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include <yaml-cpp/yaml.h>

class Imu;
class Map;
class Matcher;
class MapPoint;
class FramePoint;
class Camera;

class KeyFrame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  KeyFrame(const std::size_t &, const double &, cv::Mat, cv::Mat,
           const YAML::Node &);
  ~KeyFrame();
  std::shared_ptr<FramePoint>
  toFramePoint(const std::shared_ptr<MapPoint> &) const;

  std::size_t numMapPoints() const;
  std::vector<std::shared_ptr<MapPoint>> mapPoints();
  cv::Mat image() const;
  cv::Mat depth() const;
  std::shared_ptr<Camera> camera() const;
  Eigen::Transform<double, 3, Eigen::Isometry> getPose();
  void setPose(const Eigen::Transform<double, 3, Eigen::Isometry> &);
  Eigen::Transform<double, 3, Eigen::Isometry> pose() const;
  std::vector<std::shared_ptr<FramePoint>> framePoints();
  std::shared_ptr<FramePoint> framePoint(const std::size_t &) const;
  std::shared_ptr<MapPoint> mapPoint(const std::size_t &);
  std::size_t id() const;
  std::vector<std::shared_ptr<Imu>> imuMeasurements() const;
  void setId(const std::size_t &);
  double timestamp();
  const Eigen::VectorXf &imageEmbedding() const;
  void imageEmbedding(const Eigen::VectorXf &embedding);
  std::vector<std::shared_ptr<FramePoint>> extractORB(cv::Mat, cv::Mat,
                                                      std::size_t = 100000);
  bool estimatePose(Eigen::Transform<double, 3, Eigen::Isometry> &);
  double evaluateError() const;
  void cameraPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &) const;
  void objectPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &) const;
  std::size_t numFramePoints() const;

private:
  cv::Mat depth_, image_;
  std::size_t id_;
  double timestamp_;
  std::shared_ptr<Map> map_;
  std::shared_ptr<Camera> camera_;
  std::shared_ptr<Matcher> matcher_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr camera_points_;
  std::vector<std::shared_ptr<FramePoint>> frame_points_;
  Eigen::VectorXf image_embedding_;
  mutable std::mutex pose_mtx_;
};