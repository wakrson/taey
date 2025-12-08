
#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <opencv2/core/types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include "types.h"

class KeyFrame;
class MapPoint;
class FramePoint;
class Camera;

class Map {
public:
  Map(const YAML::Node &);
  ~Map();
  void update(const gtsam::Values &);
  std::size_t numKeyFrames() const;
  std::size_t numMapPoints() const;
  std::shared_ptr<MapPoint> mapPoint(const std::size_t &) const;
  std::shared_ptr<KeyFrame> keyFrame(const std::size_t &) const;
  std::vector<std::shared_ptr<MapPoint>> mapPoints();
  std::vector<std::shared_ptr<KeyFrame>> keyFrames();
  bool exists(const std::shared_ptr<KeyFrame> &) const;
  bool exists(const std::shared_ptr<MapPoint> &) const;
  void insert(const std::shared_ptr<KeyFrame> &);
  void insert(const std::vector<std::shared_ptr<MapPoint>> &);
  std::vector<std::shared_ptr<MapPoint>>
  track(const std::shared_ptr<KeyFrame> &) const;
  double evaluateError() const;
  void runBackend();
  std::vector<std::shared_ptr<KeyFrame>>
  findNearestKeyframes(const std::shared_ptr<KeyFrame> &key_frame,
                       const std::size_t &num_neighbors) const;

private:
  faiss::IndexFlatL2 index_;
  std::vector<Eigen::VectorXf> key_frame_embeddings_;

  YAML::Node config_;
  bool backend_running_;
  std::thread backend_thread_;
  std::queue<std::shared_ptr<KeyFrame>> key_frame_queue_;
  std::map<std::size_t, std::shared_ptr<MapPoint>> map_points_;
  std::map<std::size_t, std::shared_ptr<KeyFrame>> key_frames_;
};
