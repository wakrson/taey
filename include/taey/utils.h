#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace utils {

template <typename T> T cv2eigen(const cv::Mat &cv_mat);
template <typename Derived>
cv::Mat eigen2cv(const Eigen::MatrixBase<Derived> &eigen_mat);

} // namespace utils
