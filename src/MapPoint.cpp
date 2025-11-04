#include "MapPoint.h"
#include "Camera.h"
#include "FramePoint.h"
#include "KeyFrame.h"

MapPoint::MapPoint(const std::size_t &id) {
  this->id_ = id;
  this->object_point_ = Eigen::Vector3d(0, 0, 0);
}

MapPoint::MapPoint(std::size_t id, const std::weak_ptr<FramePoint> &frame_point,
                   const Eigen::Vector3d &object_point) {
  this->id_ = id;
  this->object_point_ = object_point;
  this->frame_points_.push_back(frame_point.lock());
}

Eigen::Vector3d MapPoint::objectPoint() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return this->object_point_;
}

void MapPoint::setObjectPoint(const Eigen::Vector3d &object_point) {
  std::lock_guard<std::mutex> lock(mtx_);
  this->object_point_ = object_point;
}

void MapPoint::setId(const std::size_t &id) { this->id_ = id; }

const std::size_t &MapPoint::id() const { return this->id_; }

std::size_t MapPoint::numFramePoints() const {
  return this->frame_points_.size();
}

std::vector<std::shared_ptr<FramePoint>> MapPoint::framePoints() {
  return this->frame_points_;
}

void MapPoint::insert(std::weak_ptr<FramePoint> frame_point) {
  this->frame_points_.push_back(frame_point.lock());
}

void MapPoint::remove(std::weak_ptr<FramePoint> frame_point) {
  std::size_t kf_id = frame_point.lock()->keyFrame()->id();

  // Find and erase the FramePoint from the vector
  auto it = std::remove_if(frame_points_.begin(), frame_points_.end(),
                           [&](const std::shared_ptr<FramePoint> &fp) {
                             return fp && fp->keyFrame()->id() == kf_id;
                           });

  if (it != frame_points_.end()) {
    frame_points_.erase(it, frame_points_.end());
  }
}