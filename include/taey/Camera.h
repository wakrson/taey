#pragma once

#include <vector>

#include <Eigen/Geometry>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

class FramePoint;

class Camera {
public:
  Camera();
  Camera(const YAML::Node &);
  Camera(Camera &);
  Camera(const Camera &);
  Eigen::MatrixXd generatePixelGrid(int height, int width, int channels) const;
  bool isInitialized() const;
  int height() const;
  int width() const;
  cv::Mat getDistCoeffs() const;
  cv::Mat getCameraMatrix() const;
  Eigen::Transform<double, 3, Eigen::Isometry> pose() const;
  void setPose(const Eigen::Transform<double, 3, Eigen::Isometry> &);
  void setPose(const Eigen::Matrix4f &);
  void setCameraMatrix(const cv::Mat &);
  void setDistCoeffs(const cv::Mat &);
  float fx() const;
  float fy() const;
  float cx() const;
  float cy() const;
  cv::Mat undistort(const cv::Mat &, int interpolation = cv::INTER_LINEAR);
  Eigen::MatrixXd backProjectToCamera(const Eigen::MatrixXd &, const Eigen::VectorXd &) const;
  Eigen::MatrixXd backProjectToWorld(const Eigen::MatrixXd &, const Eigen::VectorXd &) const;
  Eigen::MatrixXd backProjectToWorld(const std::shared_ptr<FramePoint> &) const;
  std::vector<std::shared_ptr<FramePoint>> extractORB(const cv::Mat &,
                                                      const cv::Mat &);
  bool pnp(const std::vector<cv::Point3d> &, const std::vector<cv::Point2d> &,
           Eigen::Transform<double, 3, Eigen::Isometry> &);
  static std::vector<std::shared_ptr<FramePoint>>
  match(const std::vector<std::shared_ptr<FramePoint>> &,
        const std::vector<std::shared_ptr<FramePoint>> &,
        const cv::Mat &mask = cv::Mat());

private:
  float scale_;
  float max_depth_;
  int height_, width_;
  float fx_, fy_, cx_, cy_;
  std::vector<float> dist_coeffs_;
  cv::Mat undistort_map1_, undistort_map2_;
  cv::Ptr<cv::cuda::ORB> orb_;
  Eigen::Transform<double, 3, Eigen::Isometry> pose_;
};