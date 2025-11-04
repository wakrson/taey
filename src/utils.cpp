#include <Eigen/Dense> // must come before OpenCV’s eigen.hpp
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp> // defines cv::cv2eigen() and cv::eigen2cv()

#include "utils.h"

namespace utils {

template <typename T> T cv2eigen(const cv::Mat &cv_mat) {
  T eigen_mat;
  cv::cv2eigen(cv_mat, eigen_mat); // copy data
  return eigen_mat;
}

template <typename Derived>
cv::Mat eigen2cv(const Eigen::MatrixBase<Derived> &eigen_mat) {
  cv::Mat cv_mat;
  cv::eigen2cv(eigen_mat, cv_mat); // copy data
  return cv_mat;
}

} // namespace utils
