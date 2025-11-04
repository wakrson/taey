#include <exception>

#include "FramePoint.h"
#include "MapPoint.h"

FramePoint::FramePoint(const cv::Point2d &pI, const cv::Point3d &pC,
                       const cv::Mat &descriptor, std::size_t id,
                       cv::Vec3b color) {
  this->id_ = id;
  this->color_ = color;
  this->image_point_ = Eigen::Vector2d(pI.x, pI.y);
  this->camera_point_ = Eigen::Vector3d(pC.x, pC.y, pC.z);
  this->descriptor_ = descriptor.clone();
}

FramePoint::FramePoint(const Eigen::Vector2d &pI, const Eigen::Vector3d &pC,
                       const cv::Mat &descriptor, std::size_t id,
                       cv::Vec3b color) {
  this->id_ = id;
  this->color_ = color;
  this->image_point_ = pI;
  this->camera_point_ = pC;
  this->descriptor_ = descriptor.clone();
}

cv::Vec3b FramePoint::color() const { return this->color_; }

void FramePoint::setMapPoint(std::weak_ptr<MapPoint> map_point) {
  this->map_point_ = map_point;
}

void FramePoint::setKeyFrame(std::weak_ptr<KeyFrame> key_frame) {
  this->key_frame_ = key_frame;
}

const Eigen::Vector3d &FramePoint::cameraPoint() const {
  return this->camera_point_;
}

Eigen::Vector2d FramePoint::imagePoint() const { return this->image_point_; }

const cv::Mat &FramePoint::descriptor() const { return this->descriptor_; }

const std::size_t &FramePoint::id() const { return this->id_; }

std::shared_ptr<KeyFrame> FramePoint::keyFrame() const {
  return this->key_frame_.lock();
}

std::shared_ptr<MapPoint> FramePoint::mapPoint() const {
  return this->map_point_.lock();
}

cv::KeyPoint FramePoint::keyPoint() const {
  return cv::KeyPoint(static_cast<float>(this->imagePoint()(0)),
                      static_cast<float>(this->imagePoint()(1)), 10);
}