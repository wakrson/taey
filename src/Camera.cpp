#include <vector>

#include <Eigen/Geometry>
#include <gtsam/geometry/Cal3_S2.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "FramePoint.h"

Camera::Camera() {
  this->scale_ = 1.0;
  this->height_ = 0;
  this->width_ = 0;
  this->dist_coeffs_.resize(5);
  std::fill(this->dist_coeffs_.begin(), this->dist_coeffs_.end(), 0);
  this->fx_ = 0;
  this->fy_ = 0;
  this->cx_ = 0;
  this->cy_ = 0;
  this->pose_ = Eigen::Transform<double, 3, Eigen::Isometry>::Identity();
};

Camera::Camera(const YAML::Node &config) {
  this->scale_ = config["depth_scale"].as<float>();
  this->height_ = config["height"].as<int>();
  this->width_ = config["width"].as<int>();

  this->fx_ = config["fx"].as<float>();
  this->fy_ = config["fy"].as<float>();
  this->cx_ = config["cx"].as<float>();
  this->cy_ = config["cy"].as<float>();

  for (const auto &val : config["distortion"]) {
    this->dist_coeffs_.push_back(val.as<float>());
  }
  this->pose_ = Eigen::Transform<double, 3, Eigen::Isometry>::Identity();
}

Camera::operator gtsam::Cal3_S2() const {
  return gtsam::Cal3_S2(this->fx(), this->fy(), 0.0, this->cx(), this->cy());
}

int Camera::height() const { return this->height_; }

int Camera::width() const { return this->width_; }

cv::Mat Camera::getCameraMatrix() const {
  return (cv::Mat_<float>(3, 3) << this->fx(), 0.0, this->cx(), 0.0, this->fy(),
          this->cy(), 0.0, 0.0, 1.0);
}

cv::Mat Camera::getDistCoeffs() const {
  return cv::Mat(this->dist_coeffs_).clone();
}

Eigen::Vector3f Camera::getRvec() const {
  Eigen::AngleAxisf aa(this->pose_.cast<float>().rotation());
  Eigen::Vector3f rvec(aa.axis()(0) * aa.angle(), aa.axis()(1) * aa.angle(),
                       aa.axis()(2) * aa.angle());
  return rvec;
}

float Camera::getScale() const { return this->scale_; }

Eigen::Transform<double, 3, Eigen::Isometry> Camera::pose() const {
  return this->pose_;
}

void Camera::setPose(const Eigen::Transform<double, 3, Eigen::Isometry> &pose) {
  this->pose_ = pose;
}

void Camera::setPose(const Eigen::Matrix4f &pose) {
  this->pose_.matrix() = pose.cast<double>();
}

void Camera::setPose(gtsam::Pose3 &pose) {
  this->pose_.matrix().block(0, 0, 3, 3) =
      pose.rotation().matrix().cast<double>();
  this->pose_.matrix().block(0, 3, 3, 1) = pose.translation().cast<double>();
}

void Camera::setRvec(const Eigen::Vector3f &rvec) {
  float angle(rvec.norm());
  Eigen::Vector3f axis(rvec.normalized());
  Eigen::AngleAxisf aa(angle, axis);
  this->pose_.matrix().block<3, 3>(0, 0) =
      Eigen::Matrix3f(aa.matrix()).cast<double>();
}

void Camera::setTranslation(const Eigen::Vector3f &tvec) {
  this->pose_.matrix().block<3, 1>(0, 3) = tvec.cast<double>();
}

cv::Mat Camera::undistort(const cv::Mat &src) {
  cv::Mat dst;
  cv::undistort(src, dst, this->getCameraMatrix(), this->getDistCoeffs());
  return dst.clone();
}

float Camera::fx() const { return this->fx_; }

float Camera::fy() const { return this->fy_; }

float Camera::cx() const { return this->cx_; }

float Camera::cy() const { return this->cy_; }

float Camera::k1() const { return this->dist_coeffs_[0]; }

float Camera::k2() const { return this->dist_coeffs_[1]; }

float Camera::p1() const { return this->dist_coeffs_[2]; }

float Camera::p2() const { return this->dist_coeffs_[3]; }

float Camera::k3() const { return this->dist_coeffs_[4]; }

void Camera::projectPoints(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) const {
  std::size_t n = cloud->points.size();

  // Empty point cloud
  if (n == 0) {
    return;
  }

  Eigen::RowVectorXf xs(n), ys(n), zs(n);

  // Populate Eigen vectors
  for (std::size_t i = 0; i < n; i++) {
    const auto &pt = cloud->points[i];
    xs(static_cast<Eigen::Index>(i)) = pt.x;
    ys(static_cast<Eigen::Index>(i)) = pt.y;
    zs(static_cast<Eigen::Index>(i)) = pt.z;
  }

  // Project to camera frame
  Eigen::MatrixXf pC(3, n);
  pC.row(0) = (xs.array() - this->cx()) * zs.array() / this->fx();
  pC.row(1) = (ys.array() - this->cy()) * zs.array() / this->fy();
  pC.row(2) = zs.array() / this->getScale();

  // Repopulate point cloud in camera frame
  for (std::size_t i = 0; i < n; i++) {
    auto &pt = cloud->points[i];
    pt.x = pC(0, static_cast<Eigen::Index>(i));
    pt.y = pC(1, static_cast<Eigen::Index>(i));
    pt.z = pC(2, static_cast<Eigen::Index>(i));
  }
}

std::vector<cv::Point2f>
Camera::projectPoints(const std::vector<cv::Point3f> &object_points) const {
  Eigen::Transform<double, 3, Eigen::Isometry> pose = this->pose();
  Eigen::AngleAxisd aa(pose.rotation());
  Eigen::Vector3d rvec_(aa.axis()(0) * aa.angle(), aa.axis()(1) * aa.angle(),
                        aa.axis()(2) * aa.angle());
  cv::Mat rvec = (cv::Mat_<double>(3, 1) << rvec_(0), rvec_(1), rvec_(2));
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << pose.translation()(0),
                  pose.translation()(1), pose.translation()(2));
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, rvec, tvec, this->getCameraMatrix(),
                    this->getDistCoeffs(), image_points);
  return image_points;
}

Eigen::MatrixXd Camera::projectPoints(const Eigen::MatrixXd &pW) const {
  // [n x 3]
  if (pW.cols() != 3) {
    throw std::runtime_error("Invalid shape: should be [n x 3]");
  }
  // Extract transformation (T_wc)
  Eigen::Transform<double, 3, Eigen::Isometry> wTc = this->pose();
  Eigen::Matrix3d rmat = wTc.rotation().transpose();
  Eigen::Vector3d tvec = -rmat * wTc.translation();
  // Transform points into the camera using (T_wc)^-1 -> T_cw
  Eigen::MatrixXd pC = (rmat * pW.transpose()).colwise() + tvec;
  // Normalize camera coordinates
  // Obtain camera intrinsics
  Eigen::Matrix3d K;
  cv::cv2eigen(this->getCameraMatrix().clone(), K);
  // Project normalized camera coordinates into image frame (pixels -> [u, v])
  Eigen::MatrixXd pI = (K * pC).colwise().hnormalized();
  return pI.transpose();
}

Eigen::MatrixXd Camera::transformPoints(const Eigen::MatrixXd &pW) const {
  Eigen::Transform<double, 3, Eigen::Isometry> Tcw = this->pose().inverse();
  Eigen::Matrix3d rmat = Tcw.rotation();
  Eigen::Vector3d tvec = Tcw.translation();
  Eigen::MatrixXd pC = (rmat * pW.transpose()).colwise() + tvec;
  return pC.transpose();
}

Eigen::MatrixXd
Camera::backProject(const std::shared_ptr<FramePoint> &frame_point,
                    Camera::Frame frame) const {
  // Extract image point
  Eigen::MatrixXd pI(1, 2);
  pI.row(0) = frame_point->imagePoint();

  // Extract depth
  Eigen::VectorXd Z(1);
  Z(0) = frame_point->cameraPoint()(2);
  return this->backProject(pI, Z, frame);
}

Eigen::MatrixXd Camera::backProject(const Eigen::MatrixXd &pI,
                                    const Eigen::VectorXd &Z,
                                    Camera::Frame frame) const {
  if (pI.cols() != 2) {
    throw std::invalid_argument("pI shape != [num_points x 2]");
  }

  Eigen::MatrixXd K;
  cv::cv2eigen(this->getCameraMatrix(), K);

  // Backproject pixels to the camera frame
  Eigen::MatrixXd rays = K.inverse() * (pI.transpose().colwise().homogeneous());
  Eigen::MatrixXd pC = rays * (Z / scale_).asDiagonal();
  if (frame == Camera::Frame::CAMERA) {
    return pC.transpose();
  }
  // pW
  if (frame == Camera::Frame::WORLD) {
    // Twc
    Eigen::Transform<double, 3, Eigen::Isometry> wTc = this->pose();
    Eigen::MatrixXd pW = (wTc.linear() * pC).colwise() + wTc.translation();
    return pW.transpose();
  }
  throw std::invalid_argument("backProject: Unknown frame type");
}

std::vector<std::shared_ptr<FramePoint>>
Camera::extractORB(const cv::Mat &image, const cv::Mat &depth) {
  // Convert image to grayscale
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  // Extract features
  cv::Mat descriptors;
  std::vector<cv::KeyPoint> key_points;
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
  sift->detectAndCompute(gray, cv::noArray(), key_points, descriptors);

  // Initialize image points and depths
  Eigen::VectorXd Z =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(key_points.size()));
  Eigen::MatrixXd pI =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(key_points.size()), 2);
  for (std::size_t i = 0; i < key_points.size(); i++) {
    pI.row(static_cast<Eigen::Index>(i))(0) = key_points[i].pt.x;
    pI.row(static_cast<Eigen::Index>(i))(1) = key_points[i].pt.y;
    Z(static_cast<Eigen::Index>(i)) = static_cast<double>(
        depth.at<float>(int(key_points[i].pt.y), int(key_points[i].pt.x)));
  }

  Eigen::MatrixXd pC = this->backProject(pI, Z, Camera::Frame::CAMERA);
  // Back-project image point (pixel) into camera frame (xyz)
  std::vector<std::shared_ptr<FramePoint>> frame_points;
  for (Eigen::Index i = 0; i < pC.rows(); i++) {
    // Filter out points with invalid depth
    if (pC.row(i)(2) > 0.0 && std::isfinite(pC.row(i)(2)) &&
        !std::isnan(pC.row(i)(2))) {
      cv::Mat descriptor = descriptors.row(int(i)).clone();
      int u = std::clamp(int(pI.row(i)(0)), 0, image.cols - 1);
      int v = std::clamp(int(pI.row(i)(1)), 0, image.rows - 1);
      cv::Vec3b color = image.at<cv::Vec3b>(v, u);
      std::shared_ptr<FramePoint> pF = std::make_shared<FramePoint>(
          pI.row(i), pC.row(i), descriptor, i, color);
      frame_points.push_back(pF);
    }
  }
  return frame_points;
}

bool Camera::pnp(const std::vector<cv::Point3d> &object_points,
                 const std::vector<cv::Point2d> &image_points,
                 Eigen::Transform<double, 3, Eigen::Isometry> &transform) {
  cv::Mat rvec, tvec;
  // Get camera matrix and convert to double
  cv::Mat camera_matrix = getCameraMatrix().clone();
  camera_matrix.convertTo(camera_matrix, CV_64FC1);

  // Get distortion coefficient and convert to double
  cv::Mat dist_coeffs = getDistCoeffs().clone();
  dist_coeffs.convertTo(dist_coeffs, CV_64FC1);

  if (object_points.size() < 10 || image_points.size() < 10) {
    return false;
  }
  std::vector<int> inliers;
  bool status = cv::solvePnPRansac(object_points, image_points, camera_matrix,
                                   dist_coeffs, rvec, tvec, false, 1000, 8.0,
                                   0.99, inliers, cv::SOLVEPNP_ITERATIVE);
  if (inliers.size() < 20)
    status = false;

  if (status == true) {
    // Convert rvec to rmat
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3d R;
    cv::cv2eigen(Rcv, R);

    // Convert CV matrices to eigen
    Eigen::Vector3d t;
    t << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);

    // Store transform
    transform.linear() = R;
    transform.translation() = t;
    transform.linear().transposeInPlace();
    transform.translation() = -transform.linear() * transform.translation();
  }
  return status;
}

std::vector<std::shared_ptr<FramePoint>> Camera::match(
    const std::vector<std::shared_ptr<FramePoint>> &query_frame_points,
    const std::vector<std::shared_ptr<FramePoint>> &train_frame_points) {
  std::vector<std::shared_ptr<FramePoint>> matched_train_points(
      query_frame_points.size(), nullptr);
  if (query_frame_points.size() == 0 || train_frame_points.size() == 0) {
    return matched_train_points;
  }
  // Grab query descriptors
  cv::Mat query_descriptors;
  std::for_each(query_frame_points.begin(), query_frame_points.end(),
                [&](const auto &frame_point) {
                  query_descriptors.push_back(frame_point->descriptor());
                });
  query_descriptors.convertTo(query_descriptors, CV_32F);

  // Grab train descriptors
  cv::Mat train_descriptors;
  std::for_each(train_frame_points.begin(), train_frame_points.end(),
                [&](const auto &frame_point) {
                  train_descriptors.push_back(frame_point->descriptor());
                });
  train_descriptors.convertTo(train_descriptors, CV_32F);

  cv::BFMatcher matcher(cv::NORM_L2, false);
  std::vector<std::vector<cv::DMatch>> matches;
  matcher.knnMatch(query_descriptors, train_descriptors, matches, 2);

  std::vector<cv::DMatch> good_matches;
  std::set<int> query_set, train_set;
  for (const auto &m : matches) {
    if (m[0].distance < 0.75 * m[1].distance) {
      // Only insert points that havent been tracked
      if (query_set.find(m[0].queryIdx) == query_set.end() &&
          train_set.find(m[0].trainIdx) == train_set.end()) {
        std::shared_ptr<FramePoint> query_framepoint =
            query_frame_points[static_cast<std::size_t>(m[0].queryIdx)];
        std::shared_ptr<FramePoint> train_framepoint =
            train_frame_points[static_cast<std::size_t>(m[0].trainIdx)];
        matched_train_points[static_cast<std::size_t>(m[0].queryIdx)] =
            train_framepoint;
        query_set.insert(m[0].queryIdx);
        train_set.insert(m[0].trainIdx);
      }
    }
  }
  return matched_train_points;
}

bool Camera::isWithinImage(const Eigen::Vector2i &pI) const {
  return (pI(0) > 0 && pI(0) < this->width_) &&
         (pI(1) > 0 && pI(1) < this->height_);
}

Eigen::MatrixXd Camera::generatePixelGrid(int height, int width,
                                          int channels) const {
  const std::size_t pixel_count = static_cast<std::size_t>(height * width);
  Eigen::MatrixXi grid(pixel_count, channels);

  // Generate linear indices [0, pixel_count)
  Eigen::ArrayXd indices =
      Eigen::ArrayXd::LinSpaced(static_cast<Eigen::Index>(pixel_count), 0.0,
                                static_cast<double>(pixel_count - 1));

  Eigen::ArrayXi x =
      indices.cast<int>().unaryExpr([width](int i) { return i % width; });
  Eigen::ArrayXi y =
      indices.cast<int>().unaryExpr([width](int i) { return i / width; });

  grid.col(0) = x.matrix();
  grid.col(1) = y.matrix();

  // Optional: fill extra channels (e.g., homogeneous 1s)
  if (channels > 2)
    grid.col(2).setOnes();

  return grid.cast<double>();
}