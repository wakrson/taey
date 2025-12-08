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

#include <QApplication>

#include "ImageEncoder.h"

class Map;
class Camera;
class KeyFrame;
class Visualizer;
class Optimizer;

class TAEY : public QApplication {
  Q_OBJECT
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TAEY(int &argc, char **argv, const YAML::Node &config);
  TAEY(const TAEY &);
  ~TAEY();
  TAEY &operator=(TAEY);
  std::shared_ptr<KeyFrame> operator()(const cv::Mat &, const cv::Mat &);
  bool loadConfig(const std::filesystem::path &);
  void reset();
  bool loadImageEncoder(const std::filesystem::path &);
  std::shared_ptr<Map> map() const;
  bool track(std::shared_ptr<KeyFrame> &);
signals:
  void keyFrameReady(std::shared_ptr<KeyFrame>);

private:
  std::shared_ptr<Optimizer> optimizer_;
  YAML::Node config_;
  std::shared_ptr<Map> map_;
  ImageEncoder image_encoder_;
  std::unique_ptr<Visualizer> vis_;
  std::thread vis_thread_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool running_;
  bool emit_pending_;
};