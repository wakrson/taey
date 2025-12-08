#pragma once

#include <vector>

#include <Eigen/Geometry>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <opencv2/opencv.hpp>
#include <pcl/common/projection_matrix.h>
#include <pcl/impl/point_types.hpp>

#include <yaml-cpp/yaml.h>

class FramePoint;

class Camera {
public:
  enum class Frame { CAMERA, IMAGE, WORLD };
  Camera();
  Camera(const YAML::Node &);
  Camera(Camera &);
  Camera(const Camera &);
  Eigen::MatrixXd generatePixelGrid(int height, int width, int channels) const;
  operator gtsam::Cal3_S2() const;
  operator std::shared_ptr<gtsam::Cal3_S2>() const;
  bool isInitialized() const;
  int height() const;
  int width() const;
  cv::Mat getDistCoeffs() const;
  cv::Mat getCameraMatrix() const;
  float getScale() const;
  Eigen::Vector3f getRvec() const;
  Eigen::Transform<double, 3, Eigen::Isometry> pose() const;
  void setPose(const Eigen::Transform<double, 3, Eigen::Isometry> &);
  void setPose(gtsam::Pose3 &);
  void setPose(const Eigen::Matrix4f &);
  void setRvec(const Eigen::Vector3f &);
  void setTranslation(const Eigen::Vector3f &);
  void setCameraMatrix(const cv::Mat &);
  void setDistCoeffs(const cv::Mat &);
  float fx() const;
  float fy() const;
  float cx() const;
  float cy() const;
  float k1() const;
  float k2() const;
  float p1() const;
  float p2() const;
  float k3() const;
  cv::Mat undistort(const cv::Mat &);
  std::vector<cv::Point3f> applyDistortion(const std::vector<cv::Point3f> &);
  Eigen::MatrixXd backProject(const Eigen::MatrixXd &, const Eigen::VectorXd &,
                              Camera::Frame) const;
  Eigen::MatrixXd backProject(const std::shared_ptr<FramePoint> &,
                              Camera::Frame) const;
  void projectPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &) const;
  std::vector<cv::Point2f>
  projectPoints(const std::vector<cv::Point3f> &) const;
  Eigen::MatrixXd projectPoints(const Eigen::MatrixXd &) const;
  Eigen::MatrixXd transformPoints(const Eigen::MatrixXd &) const;
  bool isWithinImage(const Eigen::Vector2i &) const;
  std::vector<std::shared_ptr<FramePoint>> extractORB(const cv::Mat &,
                                                      const cv::Mat &);
  bool pnp(const std::vector<cv::Point3d> &, const std::vector<cv::Point2d> &,
           Eigen::Transform<double, 3, Eigen::Isometry> &);
  static std::vector<std::shared_ptr<FramePoint>>
  match(const std::vector<std::shared_ptr<FramePoint>> &,
        const std::vector<std::shared_ptr<FramePoint>> &);

private:
  float scale_;
  int height_, width_;
  float fx_, fy_, cx_, cy_;
  std::vector<float> dist_coeffs_;
  Eigen::Transform<double, 3, Eigen::Isometry> pose_;
};