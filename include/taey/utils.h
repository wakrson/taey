#pragma once
/**
 * @file utils.h
 * @brief Utility functions for OpenCV ↔ Eigen conversions.
 */

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace utils {

/**
 * @brief Convert cv::Mat to Eigen matrix (copy).
 * @tparam T Eigen matrix type, e.g. Eigen::MatrixXd, Eigen::MatrixXf,
 * Eigen::Vector3d, etc.
 */
template <typename T> T cv2eigen(const cv::Mat &cv_mat);

/**
 * @brief Convert Eigen matrix to cv::Mat (copy).
 * @tparam Derived any Eigen::MatrixBase-derived type.
 */
template <typename Derived>
cv::Mat eigen2cv(const Eigen::MatrixBase<Derived> &eigen_mat);

} // namespace utils
