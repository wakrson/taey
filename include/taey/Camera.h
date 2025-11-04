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

/**
 * Pinhole camera class
 * Performs common camera functionalities
 */
class Camera {
public:
  /**
   * @brief Default camera constructor
   */
  enum class Frame { CAMERA, IMAGE, WORLD };
  Camera();
  Camera(const YAML::Node &);
  /**
   * @brief Copy constructor
   */
  Camera(Camera &);
  /**
   * @brief Const copy constructor
   */
  Camera(const Camera &);
  Eigen::MatrixXd generatePixelGrid(int height, int width, int channels) const;
  /**
   * @brief Custom cast from Camera to a GTSAM camera
   */
  operator gtsam::Cal3_S2() const;
  /**
   * @brief Custom cast from Camera to a shared pointer of GTSAM camera
   */
  operator std::shared_ptr<gtsam::Cal3_S2>() const;
  /**
   * @brief Check if camera has been initialize
   * @returns boolean
   */
  bool isInitialized() const;
  /**
   * @brief Set the camera pose using gtsam 3D pose object
   */
  /**
   * @brief Return pose as Eigen transform
   * @return Eigen transform
   */

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
  /**
   * @brief Get the camera's horizontal focal length
   * @return Focal length
   */
  float fx() const;
  /**
   * @brief Get the camera's vertical focal length
   * @return Focal length
   */
  float fy() const;
  /**
   * @brief Get the camera's vertical optical center
   * @return Optical center
   */
  float cx() const;
  /**
   * @brief Get the camera's horizontal optical center
   * @return Optical center
   */
  float cy() const;
  /**
   * @brief Get the camera's Distortion coefficient
   * @return Distortion coefficient
   */
  float k1() const;
  /**
   * @brief Get the camera's Distortion coefficient
   * @return Distortion coefficient
   */
  float k2() const;
  /**
   * @brief Get the camera's Distortion coefficient
   * @return Distortion coefficient
   */
  float p1() const;
  /**
   * @brief Get the camera's Distortion coefficient
   * @return Distortion coefficient
   */
  float p2() const;
  /**
   * @brief Get the camera's Distortion coefficient
   * @return Distortion coefficient
   */
  float k3() const;
  /**
   * @brief Undistort inputted image
   * @return Undistorted image
   */
  cv::Mat undistort(const cv::Mat &);
  /**
   * @brief Apply undistortion to camera points
   * @return Distorted camera points
   */
  std::vector<cv::Point3f> applyDistortion(const std::vector<cv::Point3f> &);
  /**
   * @brief Project 3D points in world frame into camera frame
   * @return Camera points
   */
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
  /**
   * @brief Extract ORB features as frame points
   * @return Vector of FramePoints
   */
  std::vector<std::shared_ptr<FramePoint>> extractORB(const cv::Mat &,
                                                      const cv::Mat &);
  /**
   * @brief Perform perspective-n-point on point correspondences
   * @return T_wc
   */
  bool pnp(const std::vector<cv::Point3d> &, const std::vector<cv::Point2d> &,
           Eigen::Transform<double, 3, Eigen::Isometry> &);
  /**
   * @brief Feature-based frame point matching
   * @return Map where the keys are the ID of the query points and the values
   * are pairs of corresponences
   */
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