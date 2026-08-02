#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "FramePoint.h"
#include "KeyFrame.h"
#include "MapPoint.h"

KeyFrame::KeyFrame(const std::size_t &id, const double &timestamp, cv::Mat image, cv::Mat depth, const YAML::Node &config) {
  id_ = id;
  timestamp_ = timestamp;
  camera_ = std::make_shared<Camera>(config);
  image_ = camera_->undistort(image).clone();
  // Nearest-neighbor for depth: never interpolate across depth discontinuities
  // or holes (would create flying-pixel artifacts along the viewing rays).
  depth_ = camera_->undistort(depth, cv::INTER_NEAREST).clone();
  // Extract on the undistorted image/depth so pixel coordinates match the
  // pinhole camera matrix used during back-projection.
  frame_points_ = camera_->extractORB(image_, depth_);
}

KeyFrame::~KeyFrame() { }

std::size_t KeyFrame::numFramePoints() const {
  return frame_points_.size();
}

std::size_t KeyFrame::numMapPoints() const {
  std::size_t num_map_points = 0;
  for (const auto &frame_point : frame_points_) {
    if (frame_point->mapPoint() != nullptr)
      num_map_points++;
  }
  return num_map_points;
}

cv::Mat KeyFrame::image() const { return image_; }

cv::Mat KeyFrame::depth() const { return depth_; }

Eigen::Transform<double, 3, Eigen::Isometry> KeyFrame::pose() const {
  std::lock_guard<std::mutex> lock(pose_mtx_);
  return camera_->pose();
}

void KeyFrame::setPose(
    const Eigen::Transform<double, 3, Eigen::Isometry> &pose) {
  std::lock_guard<std::mutex> lock(pose_mtx_);
  camera_->setPose(pose);
}

std::shared_ptr<Camera> KeyFrame::camera() const { return camera_; }

std::vector<std::shared_ptr<FramePoint>> KeyFrame::framePoints() {
  return frame_points_;
}

double KeyFrame::timestamp() { return timestamp_; }

const Eigen::VectorXf &KeyFrame::imageEmbedding() const {
  return image_embedding_;
}

// Setter
void KeyFrame::imageEmbedding(const Eigen::VectorXf &embedding) {
  image_embedding_ = embedding;
}

std::size_t KeyFrame::id() const { return id_; }

bool KeyFrame::estimatePose(
    Eigen::Transform<double, 3, Eigen::Isometry> &transform) {
  // Grab raw object and image points
  std::vector<cv::Point2d> image_points;
  std::vector<cv::Point3d> object_points;
  // Only return if map point has been tracked
  for (std::shared_ptr<FramePoint> frame_point : frame_points_) {
    // Check if the frame point has been tracked
    if (frame_point->mapPoint() != nullptr) {
      Eigen::Vector2d pI = frame_point->imagePoint();
      Eigen::Vector3d pW = frame_point->mapPoint()->objectPoint();
      image_points.push_back(cv::Point2d(pI(0), pI(1)));
      object_points.push_back(cv::Point3d(pW(0), pW(1), pW(2)));
    }
  }
  return camera_->pnp(object_points, image_points, transform);
}

void KeyFrame::objectPoints(Eigen::MatrixXd &cloud) const {
  // Generate pixels
  int height = camera_->height();
  int width = camera_->width();
  Eigen::MatrixXd pI = camera_->generatePixelGrid(height, width, 2);
  // Get depths
  Eigen::VectorXd Z(pI.rows());
  for (Eigen::Index i = 0; i < Z.size(); ++i) {
    int u = std::clamp(int(pI(i, 0)), 0, width - 1);
    int v = std::clamp(int(pI(i, 1)), 0, height - 1);
    Z(i) = depth_.at<float>(v, u);
  }
  // Ensure no other thread accessing pose info
  std::lock_guard<std::mutex> lock(pose_mtx_);

  // Backproject pixels into the camera frame
  Eigen::MatrixXd pW = camera_->backProjectToWorld(pI, Z);
  // Count valid points: positive input depth and finite world coordinate.
  // (pW is in the world frame, so filter on the source depth Z, not pW.z.)
  Eigen::Index num_rows = 0;
  for (Eigen::Index i = 0; i < pW.rows(); i++) {
    if (Z(i) > 0 && std::isfinite(pW.row(i)(2))) {
      ++num_rows;
    }
  }
  cloud.setZero(num_rows, 6);
  Eigen::Index idx {0};
  for (Eigen::Index i = 0; i < pW.rows(); i++) {
    if (Z(i) > 0 && std::isfinite(pW.row(i)(2))) {
      int u = std::clamp(int(pI(i, 0)), 0, width - 1);
      int v = std::clamp(int(pI(i, 1)), 0, height - 1);
      cv::Vec3b color = image_.at<cv::Vec3b>(v, u);
      cloud.row(idx)(0) = static_cast<float>(pW.row(i)(0));
      cloud.row(idx)(1) = static_cast<float>(pW.row(i)(1));
      cloud.row(idx)(2) = static_cast<float>(pW.row(i)(2));
      cloud.row(idx)(3) = color[0]; // blue
      cloud.row(idx)(4) = color[1]; // green
      cloud.row(idx)(5) = color[2]; // red
      idx++;
    }
  }
}

std::vector<std::shared_ptr<MapPoint>> KeyFrame::mapPoints() {
  std::vector<std::shared_ptr<MapPoint>> map_points;
  for (const auto &fp : frame_points_) {
    map_points.push_back(fp->mapPoint());
  }
  return map_points;
}