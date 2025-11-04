#pragma once

#include <iostream>
#include <map>
#include <stdint.h>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class MapPoint;
class KeyFrame;

class FramePoint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FramePoint(const cv::Point2d &, const cv::Point3d &, const cv::Mat &,
             std::size_t = 0, cv::Vec3b = cv::Vec3b(0, 0, 0));
  FramePoint(const Eigen::Vector2d &, const Eigen::Vector3d &, const cv::Mat &,
             std::size_t = 0, cv::Vec3b = cv::Vec3b(0, 0, 0));
  cv::Vec3b color() const;
  cv::KeyPoint keyPoint() const;
  const Eigen::Vector3d &cameraPoint() const;
  Eigen::Vector2d imagePoint() const;
  const cv::Mat &descriptor() const;
  const std::size_t &id() const;
  const cv::Point3f &objectPoint() const;
  std::shared_ptr<MapPoint> mapPoint() const;
  std::shared_ptr<KeyFrame> keyFrame() const;

  void setId(const std::size_t &);
  void setMapPoint(std::weak_ptr<MapPoint>);
  void setKeyFrame(std::weak_ptr<KeyFrame>);

private:
  std::size_t id_;
  cv::Vec3b color_;
  cv::Mat descriptor_;
  Eigen::Vector2d image_point_;
  Eigen::Vector3d camera_point_;
  std::weak_ptr<MapPoint> map_point_;
  std::weak_ptr<KeyFrame> key_frame_;
};