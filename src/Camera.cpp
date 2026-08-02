#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "FramePoint.h"

Camera::Camera() {
  scale_ = 1.0;
  max_depth_ = std::numeric_limits<float>::max();
  height_ = 0;
  width_ = 0;
  dist_coeffs_.resize(5);
  std::fill(dist_coeffs_.begin(), dist_coeffs_.end(), 0);
  fx_ = 0;
  fy_ = 0;
  cx_ = 0;
  cy_ = 0;
  pose_ = Eigen::Transform<double, 3, Eigen::Isometry>::Identity();
};

Camera::Camera(const YAML::Node &config) {
  scale_ = config["depth_scale"].as<float>();
  // Discard back-projected points beyond this metric depth (unreliable sensor
  // range). Absent from config => no clamp.
  max_depth_ = config["max_depth"] ? config["max_depth"].as<float>()
                                   : std::numeric_limits<float>::max();
  height_ = config["height"].as<int>();
  width_ = config["width"].as<int>();

  fx_ = config["fx"].as<float>();
  fy_ = config["fy"].as<float>();
  cx_ = config["cx"].as<float>();
  cy_ = config["cy"].as<float>();

  // Fixed size of 5 coefficients; missing entries mean zero distortion.
  dist_coeffs_.assign(5, 0.0f);
  std::size_t coeff_idx = 0;
  for (const auto &val : config["distortion"]) {
    if (coeff_idx >= dist_coeffs_.size()) {
      break;
    }
    dist_coeffs_[coeff_idx++] = val.as<float>();
  }
  pose_ = Eigen::Transform<double, 3, Eigen::Isometry>::Identity();
}

int Camera::height() const { return height_; }

int Camera::width() const { return width_; }

cv::Mat Camera::getCameraMatrix() const {
  return (cv::Mat_<float>(3, 3) << fx(), 0.0, cx(), 0.0, fy(),
          cy(), 0.0, 0.0, 1.0);
}

cv::Mat Camera::getDistCoeffs() const {
  return cv::Mat(dist_coeffs_).clone();
}

Eigen::Transform<double, 3, Eigen::Isometry> Camera::pose() const {
  return pose_;
}

void Camera::setPose(const Eigen::Transform<double, 3, Eigen::Isometry> &pose) {
  pose_ = pose;
}

void Camera::setPose(const Eigen::Matrix4f &pose) {
  pose_.matrix() = pose.cast<double>();
}


cv::Mat Camera::undistort(const cv::Mat &src, int interpolation) {
  // Maps are built once; callers pass INTER_NEAREST for depth.
  if (undistort_map1_.empty() || undistort_map1_.size() != src.size()) {
    cv::initUndistortRectifyMap(getCameraMatrix(), getDistCoeffs(), cv::Mat(),
                                getCameraMatrix(), src.size(), CV_16SC2,
                                undistort_map1_, undistort_map2_);
  }
  cv::Mat dst;
  cv::remap(src, dst, undistort_map1_, undistort_map2_, interpolation);
  return dst;
}

float Camera::fx() const { return fx_; }

float Camera::fy() const { return fy_; }

float Camera::cx() const { return cx_; }

float Camera::cy() const { return cy_; }

Eigen::MatrixXd Camera::backProjectToCamera(const Eigen::MatrixXd &pI,
                                            const Eigen::VectorXd &Z) const {
  if (pI.cols() != 2) {
    throw std::invalid_argument("pI shape != [num_points x 2]");
  }

  Eigen::MatrixXd K;
  cv::cv2eigen(getCameraMatrix(), K);

  // Backproject pixels to the camera frame
  Eigen::MatrixXd rays = K.inverse() * (pI.transpose().colwise().homogeneous());
  Eigen::MatrixXd pC = rays * (Z / scale_).asDiagonal();
  return pC.transpose();
}

Eigen::MatrixXd Camera::backProjectToWorld(const Eigen::MatrixXd &pI,
                                           const Eigen::VectorXd &Z) const {
  // pC: [num_points x 3]
  Eigen::MatrixXd pC = backProjectToCamera(pI, Z);
  // pW = R_wc * pC + t_wc
  Eigen::Transform<double, 3, Eigen::Isometry> wTc = pose();
  Eigen::MatrixXd pW =
      (wTc.linear() * pC.transpose()).colwise() + wTc.translation();
  return pW.transpose();
}

Eigen::MatrixXd
Camera::backProjectToWorld(const std::shared_ptr<FramePoint> &frame_point) const {
  // The FramePoint already holds the metric camera point, so transform it
  // directly rather than re-projecting from the pixel (which would re-apply
  // the depth scale).
  const Eigen::Vector3d pC = frame_point->cameraPoint();
  // pW = R_wc * pC + t_wc
  return (pose() * pC).transpose();
}

// Depth samples straddling an object boundary are unreliable (flying pixels):
// reject a sample whose 3x3 neighborhood spread exceeds a fraction of its own
// depth. Zero-depth (hole) neighbors are ignored.
static bool depthIsStable(const cv::Mat &depth, int u, int v) {
  const float d = depth.at<float>(v, u);
  if (!(d > 0)) {
    return false;
  }
  for (int dv = -1; dv <= 1; dv++) {
    for (int du = -1; du <= 1; du++) {
      const int un = std::clamp(u + du, 0, depth.cols - 1);
      const int vn = std::clamp(v + dv, 0, depth.rows - 1);
      const float dn = depth.at<float>(vn, un);
      if (dn > 0 && std::abs(dn - d) > 0.05f * d) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::shared_ptr<FramePoint>>
Camera::extractORB(const cv::Mat &image, const cv::Mat &depth) {
  // Convert image to grayscale
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  // Extract features on the GPU. Unlike the CPU version, cuda::ORB skips the
  // pre-descriptor blur by default, which badly degrades descriptor quality;
  // the trailing true re-enables it.
  if (orb_.empty()) {
    orb_ = cv::cuda::ORB::create(2000, 1.2f, 8, 31, 0, 2,
                                 cv::ORB::HARRIS_SCORE, 31, 20, true);
  }
  cv::cuda::GpuMat d_gray(gray);
  cv::cuda::GpuMat d_keypoints, d_descriptors;
  orb_->detectAndComputeAsync(d_gray, cv::noArray(), d_keypoints, d_descriptors);
  std::vector<cv::KeyPoint> key_points;
  cv::Mat descriptors;
  // Converting/downloading empty GpuMats asserts (e.g. a textureless frame).
  if (!d_keypoints.empty()) {
    orb_->convert(d_keypoints, key_points);
    d_descriptors.download(descriptors);
  }

  // Keep only the strongest keypoint per pixel (pyramid levels can repeat).
  std::map<std::pair<int, int>, int> strongest;
  for (int i = 0; i < static_cast<int>(key_points.size()); i++) {
    const auto px = std::make_pair(cvRound(key_points[static_cast<std::size_t>(i)].pt.x),
                                   cvRound(key_points[static_cast<std::size_t>(i)].pt.y));
    const auto it = strongest.find(px);
    if (it == strongest.end() ||
        key_points[static_cast<std::size_t>(i)].response >
            key_points[static_cast<std::size_t>(it->second)].response) {
      strongest[px] = i;
    }
  }
  std::vector<cv::KeyPoint> unique_key_points;
  cv::Mat unique_descriptors;
  unique_key_points.reserve(strongest.size());
  for (const auto &kv : strongest) {
    unique_key_points.push_back(key_points[static_cast<std::size_t>(kv.second)]);
    unique_descriptors.push_back(descriptors.row(kv.second));
  }
  key_points = std::move(unique_key_points);
  descriptors = unique_descriptors;

  // Initialize image points and depths
  Eigen::VectorXd Z = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(key_points.size()));
  Eigen::MatrixXd pI = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(key_points.size()), 2);

  for (std::size_t i = 0; i < key_points.size(); i++) {
    pI.row(static_cast<Eigen::Index>(i))(0) = key_points[i].pt.x;
    pI.row(static_cast<Eigen::Index>(i))(1) = key_points[i].pt.y;
    // Round (not truncate) the subpixel location to the nearest depth sample.
    const int u = std::clamp(cvRound(key_points[i].pt.x), 0, depth.cols - 1);
    const int v = std::clamp(cvRound(key_points[i].pt.y), 0, depth.rows - 1);
    // Zero marks the sample invalid; the depth filter below then drops it.
    Z(static_cast<Eigen::Index>(i)) =
        depthIsStable(depth, u, v) ? static_cast<double>(depth.at<float>(v, u))
                                   : 0.0;
  }

  Eigen::MatrixXd pC = backProjectToCamera(pI, Z);

  // Back-project image point (pixel) into camera frame (xyz)
  std::vector<std::shared_ptr<FramePoint>> frame_points;
  for (Eigen::Index i = 0; i < pC.rows(); i++) {
    // Filter out points with invalid depth
    if (pC.row(i)(2) > 0.001f && pC.row(i)(2) < max_depth_ &&
        std::isfinite(pC.row(i)(2)) && !std::isnan(pC.row(i)(2))) {
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

  if (image_points.size() != object_points.size() || 
    image_points.size() < 10) {
    return false;
  }

  // Image points are already undistorted, so pass zero distortion.
  cv::Mat camera_matrix;
  getCameraMatrix().convertTo(camera_matrix, CV_64FC1);
  cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64FC1);

  cv::Mat rvec, tvec;
  std::vector<int> inliers;
  bool status = cv::solvePnPRansac(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    rvec,
    tvec,
    false,
    10000,
    4.0,
    0.999,
    inliers,
    cv::SOLVEPNP_SQPNP
  );

  if (!status || inliers.size() < 10)
    return false;

  std::vector<cv::Point3d> inlier_object_points;
  std::vector<cv::Point2d> inlier_image_points;
  inlier_object_points.reserve(inliers.size());
  inlier_image_points.reserve(inliers.size());
  
  for (int idx : inliers) {
    inlier_object_points.push_back(object_points[static_cast<std::size_t>(idx)]);
    inlier_image_points.push_back(image_points[static_cast<std::size_t>(idx)]);
  }
  
  cv::solvePnPRefineLM(inlier_object_points, inlier_image_points,
                       camera_matrix, dist_coeffs, rvec, tvec);

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
  return status;
}

std::vector<std::shared_ptr<FramePoint>> Camera::match(
    const std::vector<std::shared_ptr<FramePoint>> &query_frame_points,
    const std::vector<std::shared_ptr<FramePoint>> &train_frame_points,
    const cv::Mat &mask) {
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

  // Grab train descriptors
  cv::Mat train_descriptors;
  std::for_each(train_frame_points.begin(), train_frame_points.end(),
                [&](const auto &frame_point) {
                  train_descriptors.push_back(frame_point->descriptor());
                });

  cv::BFMatcher matcher(cv::NORM_HAMMING, false);
  std::vector<std::vector<cv::DMatch>> matches;
  matcher.knnMatch(query_descriptors, train_descriptors, matches, 2, mask);

  std::vector<cv::DMatch> good_matches;
  std::set<int> query_set, train_set;
  // A lone candidate (under a mask) passes an absolute ORB Hamming distance
  // instead.
  for (const auto &m : matches) {
    if (m.empty()) {
      continue;
    }
    const bool good = m.size() < 2
                          ? m[0].distance < 64.0f
                          : m[0].distance < 0.75 * m[1].distance;
    if (good) {
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