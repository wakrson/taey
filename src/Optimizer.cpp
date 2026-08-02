#include "Optimizer.h"
#include "KeyFrame.h"
#include "Map.h"

using gtsam::symbol_shorthand::X;

Optimizer::Optimizer(const std::shared_ptr<Map> &map)
    : map_(map) {
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = 0.01;
  params.relinearizeSkip = 10;
  params.enableDetailedResults = false;

  isam_ = std::make_unique<gtsam::ISAM2>(params);
}

void Optimizer::update(const std::size_t &kfid) {
  std::shared_ptr<KeyFrame> kf = map_->keyFrame(kfid);

  // T_wc
  gtsam::Pose3 T_wc = gtsam::Pose3(kf->pose().matrix());

  // Add prior for the first pose we ever see (anchors the gauge). Anchor it to
  // this keyframe's own key/pose rather than assuming the first update is X(0).
  if (poses_.empty()) {
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3).finished());
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(kfid), T_wc, prior_noise));
  }

  // Add pose
  if (poses_.count(X(kfid)) == 0) {
    initial_estimate_.insert(X(kfid), T_wc);
    poses_.insert(X(kfid));
  }

  // Add between factor
  if (kfid > 0) {
    const Eigen::Isometry3d T_odom = map_->keyFrame(kfid - 1)->pose().inverse() * kf->pose();
    const gtsam::Pose3 odom(T_odom.matrix());

    auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.05, 0.05, 0.05, // roll, pitch, yaw
         0.10, 0.10, 0.10                      // x, y, z
         ).finished());

    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(kfid - 1), X(kfid), odom, odom_noise));
  }

  // Batch ISAM2 updates every 5 keyframes; factors accumulate in graph_
  // between flushes and loop closures still flush immediately.
  if (kfid % 5 == 0) {
    flush();
  }
}

void Optimizer::flush(bool force_relinearize) {
  if (graph_.empty() && initial_estimate_.empty()) {
    return;
  }
  // Run optimizer
  gtsam::ISAM2UpdateParams params;
  params.force_relinearize = force_relinearize;
  isam_->update(graph_, initial_estimate_, params);
  graph_.resize(0);
  initial_estimate_.clear();

  map_->update(isam_->calculateEstimate());
}

void Optimizer::addLoopClosure(
    const std::size_t &from, const std::size_t &to,
    const Eigen::Transform<double, 3, Eigen::Isometry> &T_rel) {
  // Both poses must already be in the graph.
  if (poses_.count(X(from)) == 0 || poses_.count(X(to)) == 0) {
    return;
  }

  // Weaker than odometry and robust, so loops nudge rather than overpower.
  auto noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 0.10, 0.10, 0.10, // roll, pitch, yaw
       0.30, 0.30, 0.30                      // x, y, z
       ).finished());
  auto huber = gtsam::noiseModel::mEstimator::Huber::Create(1.345);
  auto robust = gtsam::noiseModel::Robust::Create(huber, noise);

  graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
      X(from), X(to), gtsam::Pose3(T_rel.matrix()), robust));
  // A closure moves the whole loop; stale linearization would warp the map.
  flush(true);
}
