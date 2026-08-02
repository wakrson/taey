#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <stdint.h>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class Map;
class CLIP;
class Camera;
class KeyFrame;
class Optimizer;

class TAEY {
public:
  TAEY(const YAML::Node &config);
  TAEY(const TAEY &);
  ~TAEY();
  TAEY &operator=(TAEY);
  std::shared_ptr<KeyFrame> operator()(const cv::Mat &, const cv::Mat &, double);
  void reset();
  std::shared_ptr<Map> map() const;
  bool loopClosed() const;
  bool track(std::shared_ptr<KeyFrame> &);

private:
  int kf_stride_, frame_count_;
  YAML::Node config_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<Map> map_;
  std::unique_ptr<CLIP> clip_;

  // Keyframe selection: skip frames above this overlap and below this
  // parallax (radians).
  float kf_max_overlap_;
  float kf_min_parallax_;
  // Neighbors retrieved from the embedding index for loop closure.
  std::size_t num_neighbors_;

  // Consecutive tracking failures; guided matching is only used when zero.
  int lost_count_;
  // Whether the most recent keyframe closed a loop.
  bool loop_closed_;
  // Loop detection is suppressed for keyframes below this id, so one revisit
  // contributes one constraint instead of one per keyframe.
  std::size_t loop_cooldown_until_;
  bool insertAnchor(std::shared_ptr<KeyFrame> &);
  void detectLoopClosure(const std::shared_ptr<KeyFrame> &);

  // Constant-velocity model from the last two keyframes.
  bool have_velocity_;
  double vel_dt_;
  Eigen::Transform<double, 3, Eigen::Isometry> vel_rel_;
};