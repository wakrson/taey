#include "Map.h"
#include "Camera.h"
#include "FramePoint.h"
#include "CLIP.h"
#include "KeyFrame.h"
#include "MapPoint.h"

Map::Map(const YAML::Node &config) : index_(512) {
  config_ = config;
}

Map::~Map() { }

void Map::update(const gtsam::Values &values) {
  // Update poses, remembering each keyframe's correction.
  std::map<std::size_t, Eigen::Isometry3d> deltas;
  for (const auto &kv : values) {
    gtsam::Symbol symbol(kv.key);
    if (symbol.chr() != 'x') {
      continue;
    }
    const gtsam::Pose3 &pose = values.at<gtsam::Pose3>(kv.key);
    auto it = key_frames_.find(symbol.index());
    if (it == key_frames_.end()) {
      continue;
    }
    const Eigen::Isometry3d T_new(pose.matrix());
    const Eigen::Isometry3d delta = T_new * it->second->pose().inverse();
    it->second->setPose(T_new);
    // Corrections below 1e-6 m / rad leave map points untouched, so no-op
    // flushes don't churn the whole map.
    if (delta.translation().norm() > 1e-6 ||
        Eigen::AngleAxisd(delta.rotation()).angle() > 1e-6) {
      deltas[symbol.index()] = delta;
    }
  }
  if (deltas.empty()) {
    return;
  }

  // Map points are not optimizer variables; move each rigidly with the
  // keyframe that first observed it.
  for (const auto &entry : map_points_) {
    const std::shared_ptr<MapPoint> &mp = entry.second;
    if (mp->framePoints().empty()) {
      continue;
    }
    const std::shared_ptr<KeyFrame> anchor = mp->framePoints().front()->keyFrame();
    if (anchor == nullptr) {
      continue;
    }
    const auto it = deltas.find(anchor->id());
    if (it != deltas.end()) {
      mp->setObjectPoint(it->second * mp->objectPoint());
    }
  }
}

std::vector<std::shared_ptr<MapPoint>>
Map::track(const std::shared_ptr<KeyFrame> &key_frame, bool guided) const {
  // Get candidate map points
  std::set<std::size_t> map_point_set;
  // Get visually neighboring key frames
  std::vector<std::shared_ptr<FramePoint>> train_frame_points;
  std::vector<Eigen::Vector3d> train_world_points;
  for (const auto &kf : findNearestKeyframes(key_frame, 10)) {
    for (const auto &mp : kf->mapPoints()) {
      if (mp == nullptr)
        continue;

      // Dont add duplicate points
      if (map_point_set.find(mp->id()) != map_point_set.end())
        continue;

      if (mp->framePoints().size() == 0) {
        continue;
      }

      train_frame_points.push_back(mp->framePoints().back());
      train_world_points.push_back(mp->objectPoint());
      map_point_set.insert(mp->id());
    }
  }

  // frame points from current frame
  std::vector<std::shared_ptr<FramePoint>> query_frame_points = key_frame->framePoints();

  // Restrict candidates to query points near their pixel projected under the
  // caller's predicted pose.
  cv::Mat mask;
  if (guided) {
    const double guided_radius = 50.0; // pixels
    const std::shared_ptr<Camera> camera = key_frame->camera();
    const Eigen::Transform<double, 3, Eigen::Isometry> T_cw =
        camera->pose().inverse();
    mask = cv::Mat::zeros(static_cast<int>(query_frame_points.size()),
                          static_cast<int>(train_frame_points.size()), CV_8U);
    for (std::size_t t = 0; t < train_frame_points.size(); t++) {
      const Eigen::Vector3d pC = T_cw * train_world_points[t];
      // Behind the predicted camera: leave the whole column masked out.
      if (pC(2) <= 0) {
        continue;
      }
      const double u = camera->fx() * pC(0) / pC(2) + camera->cx();
      const double v = camera->fy() * pC(1) / pC(2) + camera->cy();
      for (std::size_t q = 0; q < query_frame_points.size(); q++) {
        const Eigen::Vector2d pI = query_frame_points[q]->imagePoint();
        const double du = pI(0) - u;
        const double dv = pI(1) - v;
        if (du * du + dv * dv < guided_radius * guided_radius) {
          mask.at<uchar>(static_cast<int>(q), static_cast<int>(t)) = 1;
        }
      }
    }
  }

  // Track query points
  std::vector<std::shared_ptr<FramePoint>> matches = Camera::match(query_frame_points, train_frame_points, mask);

  // Get map points
  std::vector<std::shared_ptr<MapPoint>> map_points(query_frame_points.size(), nullptr);

  // Set the map point if there was a match
  std::set<std::size_t> set;
  std::size_t match_cnt{0};
  for (std::size_t i = 0; i < query_frame_points.size(); i++) {
    if (matches[i] != nullptr) {
      std::size_t mid = matches[i]->mapPoint()->id();
      if (set.find(mid) != set.end()) {
        std::cerr << "Duplicate insert : " << mid << std::endl;
      }
      map_points[i] = matches[i]->mapPoint();
      match_cnt++;
      set.insert(mid);
    }
  }
  return map_points;
}

void Map::fuse(const std::shared_ptr<KeyFrame> &key_frame) {
  const double fuse_radius = 4.0;       // pixels
  const float fuse_max_distance = 50.0f; // ORB Hamming, stricter than matching:
                                         // a wrong merge corrupts the map for good

  // Map points already observed by this keyframe (also dedups candidates).
  std::set<std::size_t> seen;
  for (const auto &mp : key_frame->mapPoints()) {
    if (mp != nullptr) {
      seen.insert(mp->id());
    }
  }

  const std::vector<std::shared_ptr<FramePoint>> frame_points =
      key_frame->framePoints();
  const std::shared_ptr<Camera> camera = key_frame->camera();
  const Eigen::Transform<double, 3, Eigen::Isometry> T_cw =
      key_frame->pose().inverse();

  // Project unobserved neighboring map points into this keyframe; a projection
  // that lands on a frame point with a matching descriptor is a duplicate of
  // that frame point's map point.
  for (const auto &kf : findNearestKeyframes(key_frame, 10)) {
    for (const auto &mp : kf->mapPoints()) {
      if (mp == nullptr || mp->framePoints().empty() ||
          !seen.insert(mp->id()).second) {
        continue;
      }
      const Eigen::Vector3d pC = T_cw * mp->objectPoint();
      if (pC(2) <= 0) {
        continue;
      }
      const double u = camera->fx() * pC(0) / pC(2) + camera->cx();
      const double v = camera->fy() * pC(1) / pC(2) + camera->cy();

      const cv::Mat descriptor = mp->framePoints().back()->descriptor();
      std::shared_ptr<FramePoint> best = nullptr;
      double best_d2 = fuse_radius * fuse_radius;
      for (const auto &fp : frame_points) {
        const Eigen::Vector2d pI = fp->imagePoint();
        const double du = pI(0) - u;
        const double dv = pI(1) - v;
        const double d2 = du * du + dv * dv;
        if (d2 >= best_d2 ||
            cv::norm(fp->descriptor(), descriptor, cv::NORM_HAMMING) >=
                fuse_max_distance) {
          continue;
        }
        best = fp;
        best_d2 = d2;
      }
      if (best == nullptr || best->mapPoint() == nullptr) {
        continue;
      }

      // Merge the frame point's map point into the candidate. This direction
      // keeps the observation order intact: the current keyframe's frame point
      // is appended last, so framePoints().back() stays the newest measurement
      // (the optimizer relies on that).
      const std::shared_ptr<MapPoint> duplicate = best->mapPoint();
      for (const auto &fp : duplicate->framePoints()) {
        fp->setMapPoint(mp);
        mp->insert(fp);
      }
      map_points_.erase(duplicate->id());
    }
  }
}

void Map::cull(const std::size_t &current_kfid) {
  // Drop map points that stopped attracting observations while still below
  // the observation count the optimizer requires; they are likely spurious
  // and would otherwise accumulate as duplicate geometry.
  for (auto it = map_points_.begin(); it != map_points_.end();) {
    const std::shared_ptr<MapPoint> &mp = it->second;
    bool stale = false;
    if (mp->numFramePoints() < 3) {
      const std::vector<std::shared_ptr<FramePoint>> fps = mp->framePoints();
      const std::shared_ptr<KeyFrame> last =
          fps.empty() ? nullptr : fps.back()->keyFrame();
      // Stale after 3 keyframes without a new observation.
      stale = last == nullptr || last->id() + 3 <= current_kfid;
    }
    it = stale ? map_points_.erase(it) : std::next(it);
  }
}

void Map::insert(const std::vector<std::shared_ptr<MapPoint>> &map_points) {
  // Insert map points
  for (const auto &map_point : map_points) {
    std::size_t mpid = map_point->id();
    if (map_points_.find(mpid) == map_points_.end()) {
      map_points_[mpid] = map_point;
    }
  }
}

void Map::insert(const std::shared_ptr<KeyFrame> &key_frame) {
  std::size_t kfid = key_frame->id();
  if (key_frames_.find(kfid) == key_frames_.end()) {
    key_frames_[kfid] = key_frame;
  } else {
    throw std::runtime_error("KeyFrame with ID " +
                             std::to_string(key_frame->id()) +
                             " already exists in the map.");
  }

  // faiss row order must mirror keyframe ids (sequential, no gaps).
  Eigen::VectorXf embedding = key_frame->imageEmbedding();
  index_.add(1, embedding.data());
}

std::shared_ptr<MapPoint> Map::mapPoint(const std::size_t &id) const {
  auto it = map_points_.find(id);
  if (it == map_points_.end()) {
    return nullptr;
  }
  return it->second;
}

std::shared_ptr<KeyFrame> Map::keyFrame(const std::size_t &id) const {
  auto it = key_frames_.find(id);
  if (it == key_frames_.end()) {
    return nullptr;
  }
  return it->second;
}

std::size_t Map::numKeyFrames() const { return key_frames_.size(); }

std::size_t Map::numMapPoints() const { return map_points_.size(); }

std::vector<std::shared_ptr<KeyFrame>>
Map::findNearestKeyframes(const std::shared_ptr<KeyFrame> &key_frame,
                          const std::size_t &num_neighbors) const {
  if (index_.ntotal == 0)
    return {};

  // Get the number of embeddings
  const faiss::idx_t k = std::min<faiss::idx_t>(
      static_cast<faiss::idx_t>(num_neighbors), index_.ntotal);

  // Create the embedding
  Eigen::VectorXf embedding = key_frame->imageEmbedding();
  if (embedding.size() == 0)
    return {};

  // Query from keyframe
  faiss::idx_t *I = new faiss::idx_t[static_cast<std::size_t>(k)];
  float *D = new float[static_cast<std::size_t>(k)];
  index_.search(1, embedding.data(), k, D, I);

  // Initialize neighbors
  std::vector<std::shared_ptr<KeyFrame>> neighbors;
  for (faiss::idx_t i = 0; i < k; i++) {
    faiss::idx_t idx = I[i];
    neighbors.push_back(key_frames_.at(static_cast<std::size_t>(idx)));
  }

  delete[] I;
  delete[] D;
  return neighbors;
}