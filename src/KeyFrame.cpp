#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "FramePoint.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "utils.h"

KeyFrame::KeyFrame(const std::size_t &id, const double &timestamp,
                   cv::Mat image, cv::Mat depth, const YAML::Node &config) {
  id_ = id;
  timestamp_ = timestamp;
  camera_ = std::make_shared<Camera>(config);
  image_ = camera_->undistort(image).clone();
  depth_ = camera_->undistort(depth).clone();
  frame_points_ = camera_->extractORB(image, depth);
}

KeyFrame::~KeyFrame() {
  for (auto &fp : frame_points_) {
    const auto &mp = fp->mapPoint();
    if (mp == nullptr || fp->keyFrame() == nullptr)
      continue;

    // Remove frame point
    mp->remove(fp);
  }
}

std::shared_ptr<FramePoint>
KeyFrame::toFramePoint(const std::shared_ptr<MapPoint> &mp) const {
  Eigen::Isometry3d T_wc = camera_->pose();
  Eigen::Vector3d pW = mp->objectPoint();
  // Transform object point into frame
  Eigen::Vector3d pC = (T_wc.rotation() * pW).colwise() + T_wc.translation();

  // Project frame point into image
  Eigen::Matrix3d K;
  cv::cv2eigen(camera_->getCameraMatrix().clone(), K);
  Eigen::MatrixXd pI = (K * pC).colwise().hnormalized();
  std::shared_ptr<FramePoint> fp =
      std::make_shared<FramePoint>(pI, pC, cv::Mat{});
  return fp;
}

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

void KeyFrame::setId(const std::size_t &id) { id_ = id; }

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

void KeyFrame::objectPoints(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) const {
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
  Eigen::MatrixXd pW = camera_->backProject(pI, Z, Camera::Frame::WORLD);
  // reset point cloud
  cloud->points.clear();
  for (Eigen::Index i = 0; i < pW.rows(); i++) {
    if (std::isfinite(pW.row(i)(2))) {
      int u = std::clamp(int(pI(i, 0)), 0, width - 1);
      int v = std::clamp(int(pI(i, 1)), 0, height - 1);
      cv::Vec3b color = image_.at<cv::Vec3b>(v, u);
      pcl::PointXYZRGB point(static_cast<float>(pW.row(i)(0)),
                             static_cast<float>(pW.row(i)(1)),
                             static_cast<float>(pW.row(i)(2)));
      point.r = color[2]; // red
      point.g = color[1]; // green
      point.b = color[0]; // blue
      cloud->points.push_back(point);
    }
  }
}

void KeyFrame::cameraPoints(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) const {
  // Generate pixels
  int height = camera_->height();
  int width = camera_->width();
  Eigen::MatrixXd pI =
      camera_->generatePixelGrid(height, width, 2).cast<double>();
  // Get depths
  Eigen::VectorXd Z(pI.rows());
  for (Eigen::Index i = 0; i < Z.size(); ++i) {
    Z(i) = depth_.at<float>(int(pI(i, 0)), int(pI(i, 1)));
  }

  std::lock_guard<std::mutex> lock(pose_mtx_);

  // Backproject pixels into the camera frame
  Eigen::MatrixXd pC = camera_->backProject(pI, Z, Camera::Frame::CAMERA);
  cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  for (Eigen::Index i = 0; i < pC.rows(); i++) {
    if (pC.row(i)(2) > 0 && std::isfinite(pC.row(i)(2))) {
      int u = std::clamp(int(pI(i, 0)), 0, width - 1);
      int v = std::clamp(int(pI(i, 1)), 0, height - 1);
      cv::Vec3b color = image_.at<cv::Vec3b>(v, u);
      pcl::PointXYZRGB point(static_cast<float>(pC.row(i)(0)),
                             static_cast<float>(pC.row(i)(1)),
                             static_cast<float>(pC.row(i)(2)));
      point.r = color[2]; // red
      point.g = color[1]; // green
      point.b = color[0]; // blue
      cloud->points.push_back(point);
    }
  }
}

std::shared_ptr<MapPoint> KeyFrame::mapPoint(const std::size_t &id) {
  // search all of the frame points
  auto it = std::find_if(frame_points_.begin(), frame_points_.end(),
                         [&](const std::shared_ptr<FramePoint> &frame_point) {
                           if (frame_point->mapPoint() != nullptr) {
                             return frame_point->mapPoint()->id() == id;
                           }
                           return false;
                         });
  if (it != frame_points_.end()) {
    return map_->mapPoint((*it)->id());
  }
  return nullptr;
}

std::vector<std::shared_ptr<MapPoint>> KeyFrame::mapPoints() {
  std::set<std::size_t> set;
  std::vector<std::shared_ptr<MapPoint>> map_points;
  for (auto fp : frame_points_) {
    std::size_t mid = fp->mapPoint()->id();
    if (set.find(mid) != set.end()) {
      std::cerr << "Duplicate insert in keyframe : " << mid << std::endl;
    }
    map_points.push_back(fp->mapPoint());
    set.insert(mid);
  }
  return map_points;
}

double KeyFrame::evaluateError() const {
  Eigen::MatrixXd pW =
      Eigen::MatrixXd::Zero(3, static_cast<Eigen::Index>(frame_points_.size()));
  Eigen::MatrixXd pI =
      Eigen::MatrixXd::Zero(3, static_cast<Eigen::Index>(frame_points_.size()));
  Eigen::Index idx(0);
  // Collect all of the FramePoints and the corresponding MapPoints
  // observed in this KeyFrame
  std::for_each(frame_points_.begin(), frame_points_.end(),
                [&](const std::shared_ptr<FramePoint> &frame_point) {
                  // Set object point
                  pW(0, idx) = frame_point->mapPoint()->objectPoint()(0);
                  pW(1, idx) = frame_point->mapPoint()->objectPoint()(1);
                  pW(2, idx) = frame_point->mapPoint()->objectPoint()(2);
                  // Set image point
                  pI(0, idx) = frame_point->imagePoint()(0);
                  pI(1, idx) = frame_point->imagePoint()(1);
                  pI(2, idx) = 1.0;
                  idx++;
                });
  // Project object points into the image frame
  // Eigen::MatrixXd projected_pI = camera()->projectPoints(pW);
  // Get the norm: error_i ​= sqrt((u_i - u^_i​)^2 + (v_i ​- v^_i)^2)
  // dims: [1 x N]
  Eigen::MatrixXd error =
      (camera()->projectPoints(pW).cast<double>() - pI.cast<double>());
  // Take the sum pixel distance
  return error.rowwise().norm().sum();
}