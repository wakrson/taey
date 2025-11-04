#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <unordered_map>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

class FramePoint;
class KeyFrame;
class Camera;

class MapPoint {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MapPoint(const std::size_t &);
  MapPoint(std::size_t, const std::weak_ptr<FramePoint> &,
           const Eigen::Vector3d &);

  void setId(const std::size_t &);
  void setObjectPoint(const Eigen::Vector3d &);
  const std::size_t &id() const;
  std::size_t &keyFrameId();
  const std::size_t &keyFrameId() const;
  Eigen::Vector3d objectPoint() const;
  std::size_t numFramePoints() const;
  std::vector<std::shared_ptr<FramePoint>> framePoints();
  void insert(std::weak_ptr<FramePoint>);
  void remove(std::weak_ptr<FramePoint>);
  void setKeyFrame(std::shared_ptr<KeyFrame> &);
  std::shared_ptr<KeyFrame> keyFrame();

private:
  mutable std::mutex mtx_;
  std::size_t id_, key_frame_id_;
  Eigen::Vector3d object_point_;
  std::shared_ptr<KeyFrame> key_frame_;
  std::vector<std::shared_ptr<FramePoint>> frame_points_;
};