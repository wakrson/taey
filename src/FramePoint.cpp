#include <exception>

#include "FramePoint.h"
#include "MapPoint.h"

FramePoint::FramePoint(const cv::Point2d &pI, const cv::Point3d &pC,
                       const cv::Mat &descriptor, std::size_t id,
                       cv::Vec3b color) {
  id_ = id;
  color_ = color;
  image_point_ = Eigen::Vector2d(pI.x, pI.y);
  camera_point_ = Eigen::Vector3d(pC.x, pC.y, pC.z);
  descriptor_ = descriptor.clone();
}

FramePoint::FramePoint(const Eigen::Vector2d &pI, const Eigen::Vector3d &pC,
                       const cv::Mat &descriptor, std::size_t id,
                       cv::Vec3b color) {
  id_ = id;
  color_ = color;
  image_point_ = pI;
  camera_point_ = pC;
  descriptor_ = descriptor.clone();
}

cv::Vec3b FramePoint::color() const { return color_; }

void FramePoint::setMapPoint(std::weak_ptr<MapPoint> map_point) {
  map_point_ = map_point;
}

void FramePoint::setKeyFrame(std::weak_ptr<KeyFrame> key_frame) {
  key_frame_ = key_frame;
}

const Eigen::Vector3d &FramePoint::cameraPoint() const {
  return camera_point_;
}

Eigen::Vector2d FramePoint::imagePoint() const { return image_point_; }

const cv::Mat &FramePoint::descriptor() const { return descriptor_; }

const std::size_t &FramePoint::id() const { return id_; }

std::shared_ptr<KeyFrame> FramePoint::keyFrame() const {
  return key_frame_.lock();
}

std::shared_ptr<MapPoint> FramePoint::mapPoint() const {
  return map_point_.lock();
}

cv::KeyPoint FramePoint::keyPoint() const {
  return cv::KeyPoint(static_cast<float>(imagePoint()(0)),
                      static_cast<float>(imagePoint()(1)), 10);
}