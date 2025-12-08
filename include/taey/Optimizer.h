#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/sam/RangeFactor.h>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <opencv2/core/eigen.hpp>

class Map;
class KeyFrame;
class Camera;
class MapPoint;

class Optimizer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Optimizer(const std::shared_ptr<Map> &, const YAML::Node &);
  void update(const std::size_t &);
  gtsam::Values optimize();
  gtsam::Values currentEstimate() const;
  static double projectionError(const gtsam::Pose3 &, const gtsam::Cal3_S2 &,
                                const gtsam::Point3 &, const gtsam::Point2 &);

private:
  bool use_projection_factors_;
  std::set<gtsam::Key> landmarks_, poses_;

  std::shared_ptr<Map> map_;
  bool prior_added_;
  YAML::Node config_;
  std::shared_ptr<gtsam::ISAM2> isam_;
  std::shared_ptr<gtsam::Cal3_S2> camera_;

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_estimate_;
  gtsam::Values current_estimate_;
  std::set<std::pair<std::size_t, std::size_t>> kvs_;
};
