#pragma once

#include <set>

#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/sam/RangeFactor.h>

class Map;

class Optimizer {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Optimizer(const std::shared_ptr<Map> &);
  void update(const std::size_t &);
  gtsam::Values currentEstimate() const;
  static double projectionError(const gtsam::Pose3 &, const gtsam::Cal3_S2 &,
                                const gtsam::Point3 &, const gtsam::Point2 &);

private:
  std::set<gtsam::Key> landmarks_, poses_;

  std::shared_ptr<Map> map_;
  std::shared_ptr<gtsam::ISAM2> isam_;

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_estimate_;
  gtsam::Values current_estimate_;
};
