#include "Optimizer.h"
#include "Camera.h"
#include "FramePoint.h"
#include "CLIP.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"

using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

double Optimizer::projectionError(const gtsam::Pose3 &Twc,
                                  const gtsam::Cal3_S2 &K,
                                  const gtsam::Point3 &Pw,
                                  const gtsam::Point2 &uv_meas) {
  try {
    gtsam::PinholeCamera<gtsam::Cal3_S2> cam(Twc.inverse(), K);
    const gtsam::Point2 uv = cam.project(Pw);
    const double err = (uv - uv_meas).norm();
    return err;
  }
  catch (const gtsam::CheiralityException&) {
    return std::numeric_limits<double>::infinity();
  }
}

Optimizer::Optimizer(const std::shared_ptr<Map> &map, const YAML::Node &config)
    : config_(config) {
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = 0.01;
  params.relinearizeSkip = 1;
  params.cacheLinearizedFactors = false;
  params.enableDetailedResults = false;

  isam_ = std::make_unique<gtsam::ISAM2>(params);
  graph_ = gtsam::NonlinearFactorGraph();
  use_projection_factors_ = true;
  map_ = map;
}

gtsam::Values Optimizer::currentEstimate() const {
  return this->current_estimate_;
}

void Optimizer::update(const std::size_t &kfid) {
  std::shared_ptr<KeyFrame> kf = map_->keyFrame(kfid);

  // T_wc
  gtsam::Pose3 T_wc = gtsam::Pose3(kf->pose().matrix());

  auto K = std::make_shared<gtsam::Cal3_S2>(static_cast<gtsam::Cal3_S2>(*kf->camera()));

  auto px = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
  auto huber = gtsam::noiseModel::mEstimator::Huber::Create(2.0);
  auto robust_px = gtsam::noiseModel::Robust::Create(huber, px);

  // Add prior for first pose
  if (poses_.size() == 0) {
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3).finished());
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), gtsam::Pose3::Identity(),
                                                prior_noise));
  }

  // Add pose
  if (poses_.count(X(kfid)) == 0) {
    initial_estimate_.insert(X(kfid), T_wc);
    poses_.insert(X(kfid));
  }

  // Add between factor
  if (kfid > 0) {
    Eigen::Isometry3d prev_pose = map_->keyFrame(kfid - 1)->pose();
    Eigen::Isometry3d curr_pose = kf->pose(); // T_wc
    Eigen::Isometry3d T_odom = prev_pose.inverse() * curr_pose;
    gtsam::Pose3 odom(T_odom.matrix());

    auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.05, 0.05, 0.05, // roll, pitch, yaw
         0.10, 0.10, 0.10                      // x, y, z
         ).finished());

    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(kfid - 1), X(kfid), odom, odom_noise));
  }

  if (use_projection_factors_) {
    // Landmarks / projections
    for (const auto &mp : kf->mapPoints()) {
      if (!mp)
        continue;

      const std::size_t lmid = mp->id();
      const gtsam::Point3 pW(mp->objectPoint());

      // landmark already been inserted... add another projection factor
      if (landmarks_.count(L(lmid)) > 0) {
        // Add most recent measurement
        const gtsam::Point2 z(mp->framePoints().back()->imagePoint());
        graph_.add(gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3,
                                                  gtsam::Cal3_S2>(
            z, robust_px, X(kfid), L(lmid), K));
      }
      // Check if we should add new landmark and projections
      else {
        // Count observations in map point
        std::vector<gtsam::RangeFactor<gtsam::Pose3, gtsam::Point3>> range_factors;
        std::vector<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>> proj_factors;
        for (const auto &fp : mp->framePoints()) {
          const auto kf_j = fp->keyFrame();
          const gtsam::Pose3 Twc_j(kf_j->pose().matrix());

          if (!kf_j)
            continue;

          const gtsam::Point2 z(fp->imagePoint());
          if (projectionError(Twc_j, *K, pW, z) >= 3.0)
              continue;

          // Range factor
          const double d = fp->cameraPoint()(2); // meters
          if (d > 0) {
            auto range_noise = gtsam::noiseModel::Isotropic::Sigma(1, std::max(0.005, 0.01 * d));
            range_factors.emplace_back(X(kf_j->id()), L(lmid), d, range_noise);
            proj_factors.emplace_back(z, robust_px, X(kf_j->id()), L(mp->id()), K);
          }
        }

        if (proj_factors.size() < 5)
          continue;

        // Add landmark prior
        if (landmarks_.empty()) {
          auto prior_noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.20);
          graph_.add(gtsam::PriorFactor<gtsam::Point3>(L(lmid), pW, prior_noise));
        }

        // Insert landmark initial estimate if missing
        if (landmarks_.count(L(lmid)) == 0) {
          initial_estimate_.insert(L(lmid), pW);
          landmarks_.insert(L(lmid));
        }
        
        // Add factors to graph
        for (const auto &factor : proj_factors) {
          graph_.add(factor);
        }

        // Add range factors
        for (const auto &factor : range_factors) {
          graph_.add(factor);
        }
      }
    }
  }

  // Update ISAM2 now; don’t leave graph in limbo
  if (!graph_.empty() || !initial_estimate_.empty()) {
    // Run optimizer
    isam_->update(graph_, initial_estimate_);
    current_estimate_ = isam_->calculateEstimate();
    graph_.resize(0);
    initial_estimate_.clear();

    // Update map
    map_->update(current_estimate_);
  }
}