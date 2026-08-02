#pragma once

#include <set>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose3.h>

class Map;

class Optimizer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Optimizer(const std::shared_ptr<Map> &);
  void update(const std::size_t &);
  void addLoopClosure(const std::size_t &, const std::size_t &,
                      const Eigen::Transform<double, 3, Eigen::Isometry> &);

private:
  void flush(bool force_relinearize = false);

  std::set<gtsam::Key> poses_;

  std::shared_ptr<Map> map_;
  std::shared_ptr<gtsam::ISAM2> isam_;

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_estimate_;
};
