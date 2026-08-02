#include <cmath>

#include "FramePoint.h"
#include "Camera.h"
#include "CLIP.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "Optimizer.h"
#include "TAEY.h"

// Scale a relative motion to a different time interval (constant velocity).
static Eigen::Isometry3d scaleMotion(const Eigen::Isometry3d &T, double s) {
  const Eigen::AngleAxisd aa(T.rotation());
  Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
  out.linear() = Eigen::AngleAxisd(aa.angle() * s, aa.axis()).toRotationMatrix();
  out.translation() = T.translation() * s;
  return out;
}

TAEY::TAEY(const YAML::Node &config) {
  config_ = config;
  lost_count_ = 0;
  have_velocity_ = false;
  vel_dt_ = 0;
  vel_rel_ = Eigen::Isometry3d::Identity();
  map_ = std::make_shared<Map>(config);
  optimizer_ = std::make_shared<Optimizer>(map_);
  clip_ = std::make_unique<CLIP>(config_["encoder"].as<std::string>());
  kf_stride_ = config_["keyframe_stride"] ? config_["keyframe_stride"].as<int>() : 1;
  frame_count_ = 0;
  loop_closed_ = false;
  loop_cooldown_until_ = 0;
  kf_max_overlap_ = config_["keyframe_max_overlap"] ? config_["keyframe_max_overlap"].as<float>() : 0.5f;
  // Configured in degrees, stored in radians.
  kf_min_parallax_ = (config_["keyframe_min_parallax"] ? config_["keyframe_min_parallax"].as<float>() : 1.0f) * static_cast<float>(EIGEN_PI / 180.0);
  num_neighbors_ = config_["num_neighbors"] ? config_["num_neighbors"].as<std::size_t>() : 20;
}

void TAEY::reset() {
  lost_count_ = 0;
  frame_count_ = 0;
  loop_closed_ = false;
  loop_cooldown_until_ = 0;
  have_velocity_ = false;
  map_ = std::make_shared<Map>(config_);
  optimizer_ = std::make_shared<Optimizer>(map_);
}

TAEY::~TAEY() { }

std::shared_ptr<Map> TAEY::map() const { return map_; }

bool TAEY::loopClosed() const { return loop_closed_; }

std::shared_ptr<KeyFrame> TAEY::operator()(const cv::Mat &image, const cv::Mat &depth, double timestamp) {
  if (frame_count_++ % kf_stride_ != 0) {
    return nullptr;
  }

  Eigen::VectorXf features = clip_->encode(image);

  std::size_t key_frame_id = map_->numKeyFrames();
  std::shared_ptr<KeyFrame> key_frame = std::make_shared<KeyFrame>(key_frame_id, timestamp, image, depth, config_);

  for (auto &fp : key_frame->framePoints()) {
    fp->setKeyFrame(key_frame);
  }

  key_frame->imageEmbedding(features);

  if (key_frame->numFramePoints() == 0) {
    return nullptr;
  }

  // Only keep successfully tracked keyframes.
  if (!track(key_frame)) {
    return nullptr;
  }

  return key_frame;
}

bool TAEY::insertAnchor(std::shared_ptr<KeyFrame> &key_frame) {
  // Populate the map from this frame alone at its current pose.
  std::size_t num_map_points = map_->numMapPoints();
  std::vector<std::shared_ptr<MapPoint>> mps(key_frame->framePoints().size(), nullptr);
  for (std::size_t i = 0; i < key_frame->framePoints().size(); i++) {
    // Backproject to get image point in world frame
    Eigen::MatrixXd pW = key_frame->camera()->backProjectToWorld(
        key_frame->framePoints()[i]);
    // Create new map point
    mps[i] = std::make_shared<MapPoint>(num_map_points + i,
                                        key_frame->framePoints()[i],
                                        pW.transpose());
    // Set frame points map point (weak link)
    key_frame->framePoints()[i]->setMapPoint(mps[i]);
  }
  map_->insert(key_frame);
  map_->insert(mps);
  optimizer_->update(key_frame->id());
  return true;
}

bool TAEY::track(std::shared_ptr<KeyFrame> &key_frame) {
  // First insert
  if (map_->numKeyFrames() == 0) {
    return insertAnchor(key_frame);
  }

  // Predict the pose from the constant-velocity model to guide matching.
  bool guided = false;
  if (have_velocity_ && lost_count_ == 0) {
    const std::shared_ptr<KeyFrame> prev_key_frame =
        map_->keyFrame(map_->numKeyFrames() - 1);
    const double dt = key_frame->timestamp() - prev_key_frame->timestamp();
    key_frame->setPose(prev_key_frame->pose() *
                       scaleMotion(vel_rel_, dt / vel_dt_));
    guided = true;
  }

  std::vector<std::shared_ptr<MapPoint>> matches = map_->track(key_frame, guided);

  int track_cnt = static_cast<int>(
      std::count_if(matches.begin(), matches.end(),
                    [](const auto &m) { return m != nullptr; }));

  std::vector<std::shared_ptr<MapPoint>> map_points(matches.size(), nullptr);
  for (std::size_t i = 0; i < matches.size(); i++) {
    if (matches[i] != nullptr) {
      key_frame->framePoints()[i]->setMapPoint(matches[i]);
      map_points[i] = matches[i];
    }
  }

  // Estimate pose with matches
  bool status = true;
  if (map_->numKeyFrames() > 0) {
    Eigen::Transform<double, 3, Eigen::Isometry> pose;
    status = key_frame->estimatePose(pose);
    if (status) {
      key_frame->setPose(pose);

      // Reject implausible motion; velocity caps scaled by elapsed time.
      const double max_trans_vel = 1.0;  // meters per second
      const double max_rot_vel = 1.0472; // radians (~60 deg) per second
      const std::shared_ptr<KeyFrame> prev_key_frame =
          map_->keyFrame(key_frame->id() - 1);
      const Eigen::Transform<double, 3, Eigen::Isometry> T_prev =
          prev_key_frame->pose();
      const Eigen::Transform<double, 3, Eigen::Isometry> T_rel =
          T_prev.inverse() * pose;
      const double d_trans = T_rel.translation().norm();
      const double d_rot = Eigen::AngleAxisd(T_rel.rotation()).angle();
      const double dt =
          std::max(key_frame->timestamp() - prev_key_frame->timestamp(), 0.5);
      if (d_trans > max_trans_vel * dt || d_rot > max_rot_vel * dt) {
        status = false;
      }

      // Skip a redundant frame: high map overlap and low parallax.
      if (status) {
        const double overlap = static_cast<double>(track_cnt) /
                               static_cast<double>(key_frame->numFramePoints());
        const Eigen::Vector3d c_prev = T_prev.translation();
        const Eigen::Vector3d c_curr = pose.translation();
        double parallax = 0;
        for (const auto &match : matches) {
          if (match == nullptr) {
            continue;
          }
          const Eigen::Vector3d pW = match->objectPoint();
          const double cos_angle = (pW - c_prev).normalized().dot((pW - c_curr).normalized());
          parallax += std::acos(std::clamp(cos_angle, -1.0, 1.0));
        }
        parallax /= track_cnt;

        if (overlap > kf_max_overlap_ && parallax < kf_min_parallax_) {
          lost_count_ = 0;
          key_frame.reset();
          return false;
        }
      }
    }
  }

  if (status) {
    std::size_t num_map_points = map_->numMapPoints();
    // Create new map points with new pose
    for (std::size_t i = 0; i < matches.size(); i++) {
      // No matching map point (create a new one)
      if (matches[i] == nullptr) {
        // Backproject to get image point in world frame
        Eigen::MatrixXd pW = key_frame->camera()->backProjectToWorld(
            key_frame->framePoints()[i]);
        // Create new map point
        map_points[i] = std::make_shared<MapPoint>(
            num_map_points, key_frame->framePoints()[i], pW.transpose());
        // Set frame points map point (weak link)
        key_frame->framePoints()[i]->setMapPoint(map_points[i]);
        num_map_points++;
      }
      // Insert FramePoint into MapPoint
      else {
        matches[i]->insert(key_frame->framePoints()[i]);
      }
    }
    // Add to map
    map_->insert(key_frame);
    map_->insert(map_points);
    // Merge duplicate map points, then drop stale under-observed ones.
    map_->fuse(key_frame);
    map_->cull(key_frame->id());
    optimizer_->update(key_frame->id());
    detectLoopClosure(key_frame);
    lost_count_ = 0;

    // Update the constant-velocity model from the last two keyframes.
    const std::shared_ptr<KeyFrame> prev_key_frame = map_->keyFrame(key_frame->id() - 1);
    vel_dt_ = key_frame->timestamp() - prev_key_frame->timestamp();
    vel_rel_ = prev_key_frame->pose().inverse() * key_frame->pose();
    have_velocity_ = vel_dt_ > 0;
  }
  // delete
  else {
    ++lost_count_;
    key_frame.reset();
  }
  return status;
}

void TAEY::detectLoopClosure(const std::shared_ptr<KeyFrame> &key_frame) {
  loop_closed_ = false;
  // Visually similar, temporally distant candidates verified with PnP.
  if (key_frame->id() < loop_cooldown_until_) {
    return;
  }

  for (const auto &candidate :
       map_->findNearestKeyframes(key_frame, num_neighbors_)) {
    // At least 50 keyframes between loop ends.
    if (candidate->id() + 50 > key_frame->id()) {
      continue;
    }
    // CLIP cosine distance.
    if (CLIP::cosineDistance(key_frame->imageEmbedding(),
                             candidate->imageEmbedding()) > 0.05) {
      continue;
    }

    std::vector<std::shared_ptr<FramePoint>> matches = Camera::match(key_frame->framePoints(), candidate->framePoints());
    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    for (std::size_t i = 0; i < matches.size(); i++) {
      if (matches[i] == nullptr || matches[i]->mapPoint() == nullptr) {
        continue;
      }
      const Eigen::Vector3d pW = matches[i]->mapPoint()->objectPoint();
      const Eigen::Vector2d pI = key_frame->framePoints()[i]->imagePoint();
      object_points.emplace_back(pW(0), pW(1), pW(2));
      image_points.emplace_back(pI(0), pI(1));
    }

    Eigen::Transform<double, 3, Eigen::Isometry> T_loop;
    // At least 40 3D-2D matches before PnP.
    if (object_points.size() < 40 ||
        !key_frame->camera()->pnp(object_points, image_points, T_loop)) {
      continue;
    }

    std::cout << "TAEY: loop closure " << key_frame->id() << " -> "
              << candidate->id() << " (" << object_points.size() << " matches)"
              << std::endl;
    optimizer_->addLoopClosure(candidate->id(), key_frame->id(),
                               candidate->pose().inverse() * T_loop);
    loop_closed_ = true;
    loop_cooldown_until_ = key_frame->id() + 15;
    // At most one loop constraint per keyframe.
    return;
  }
}