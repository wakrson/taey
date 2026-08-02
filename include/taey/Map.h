
#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <opencv2/core/types.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

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
  void insert(const std::shared_ptr<KeyFrame> &);
  void insert(const std::vector<std::shared_ptr<MapPoint>> &);
  std::vector<std::shared_ptr<MapPoint>>
  track(const std::shared_ptr<KeyFrame> &, bool guided = false) const;
  void fuse(const std::shared_ptr<KeyFrame> &);
  void cull(const std::size_t &current_kfid);
  std::vector<std::shared_ptr<KeyFrame>>
  findNearestKeyframes(const std::shared_ptr<KeyFrame> &key_frame,
                       const std::size_t &num_neighbors) const;

private:
  faiss::IndexFlatL2 index_;

  YAML::Node config_;
  std::map<std::size_t, std::shared_ptr<MapPoint>> map_points_;
  std::map<std::size_t, std::shared_ptr<KeyFrame>> key_frames_;
};
