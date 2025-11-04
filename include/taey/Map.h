
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
// class Optimizer;
class Camera;

class Map {
public:
  /**
   * @brief Default constructor
   */
  Map(const YAML::Node &);
  /**
   * @brief Default Destructor
   */
  ~Map();
  /*
   * @brief Access the graph optimizer
   * @return std::shared_ptr<Optimizer>
   */
  // std::shared_ptr<Optimizer> optimizer() const;
  /**
   * @brief Update map variables with GTSAM values
   * @param values: GTSAM values object
   * @return Bool
   */
  void update(const gtsam::Values &);
  std::size_t numKeyFrames() const;
  std::size_t numMapPoints() const;
  /**
   * @brief Get map point by ID
   * @param id: Map point ID
   * @return Shared_ptr to map point
   */
  std::shared_ptr<MapPoint> mapPoint(const std::size_t &) const;
  /**
   * @brief Get KeyFrame by ID
   * @param id: KeyFrame ID
   * @return Shared_ptr to KeyFrame
   */
  std::shared_ptr<KeyFrame> keyFrame(const std::size_t &) const;
  /**
   * @brief Get the map points currently in the map
   * @return Vector of shared_ptr map points
   */
  std::vector<std::shared_ptr<MapPoint>> mapPoints();
  /**
   * @brief Get the key frames currently in the map
   * @return Vector of shared_ptr key frames
   */
  std::vector<std::shared_ptr<KeyFrame>> keyFrames();
  /**
   * @brief Check to see if key frame has been inserted
   * @param kf: KeyFrame
   * @return Flag that specifies whether key frame exists
   */
  bool exists(const std::shared_ptr<KeyFrame> &) const;
  bool exists(const std::shared_ptr<MapPoint> &) const;

  /**
   * @brief Add KeyFrame to map
   * @param kf: KeyFrame to insert
   * @return ID of inserted KeyFrame
   */
  void insert(const std::shared_ptr<KeyFrame> &);
  void insert(const std::vector<std::shared_ptr<MapPoint>> &);
  /**
   * @brief Track the frame points to framepoints present in the map
   * @param fps: Vector of FramePoints
   * @param window_size: Number of KeyFrames to query
   * @return Map with query FramePoint index as key and mapped FramePoint as
   * value
   */
  std::vector<std::shared_ptr<MapPoint>>
  track(const std::shared_ptr<KeyFrame> &) const;
  /**
   * @brief Calculate the total reprojection error for the entire map
   * @return Summed reprojection error
   */
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
  // std::shared_ptr<Optimizer> optimizer_;
  std::queue<std::shared_ptr<KeyFrame>> key_frame_queue_;
  std::map<std::size_t, std::shared_ptr<MapPoint>> map_points_;
  std::map<std::size_t, std::shared_ptr<KeyFrame>> key_frames_;
};
