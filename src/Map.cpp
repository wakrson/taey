#include "Map.h"
#include "Camera.h"
#include "FramePoint.h"
#include "ImageEncoder.h"
#include "KeyFrame.h"
#include "MapPoint.h"

Map::Map(const YAML::Node &config) : index_(512) {
  config_ = config;
}

Map::~Map() { }

bool Map::exists(const std::shared_ptr<KeyFrame> &key_frame) const {
  const auto id = key_frame->id();
  const auto it = key_frames_.find(id);

  const bool found = (it != key_frames_.end());
  return found;
}

bool Map::exists(const std::shared_ptr<MapPoint> &map_point) const {
  const auto id = map_point->id();
  const auto it = map_points_.find(id);

  return (it != map_points_.end());
}

void Map::update(const gtsam::Values &values) {
  // Update poses
  for (const auto &kv : values) {
    const gtsam::Key key = kv.key;
    gtsam::Symbol symbol(key);

    if (symbol.chr() == 'x') {
      const gtsam::Pose3 &pose = values.at<gtsam::Pose3>(key);
      auto it = key_frames_.find(symbol.index());
      it->second->setPose(Eigen::Isometry3d(pose.matrix()));

    } else if (symbol.chr() == 'l') {
      const gtsam::Point3 &pW = values.at<gtsam::Point3>(key);
      auto it = map_points_.find(symbol.index());
      it->second->setObjectPoint(pW);
    }
  }
}

std::vector<std::shared_ptr<MapPoint>>
Map::track(const std::shared_ptr<KeyFrame> &key_frame) const {
  // Get candidate map points
  std::set<std::size_t> map_point_set;
  // Get visually neighboring key frames
  std::vector<std::shared_ptr<FramePoint>> train_frame_points;
  for (const auto &kf : findNearestKeyframes(key_frame, 40)) {
    for (const auto &mp : kf->mapPoints()) {
      // Dont add duplicate points
      if (map_point_set.find(mp->id()) != map_point_set.end())
        continue;

      if (mp->framePoints().size() == 0) {
        continue;
      }

      train_frame_points.push_back(mp->framePoints().back());
      map_point_set.insert(mp->id());
    }
  }

  // frame points from current frame
  std::vector<std::shared_ptr<FramePoint>> query_frame_points = key_frame->framePoints();

  // Track query points
  std::vector<std::shared_ptr<FramePoint>> matches =
      Camera::match(query_frame_points, train_frame_points);

  // Get average distance between framepoints: [query_frame_points] x [matches]
  double pixel_dist = 0;
  for (std::size_t i = 0; i < query_frame_points.size(); i++) {
    if (matches[i] != nullptr) {
      const auto &fp1 = query_frame_points[i];
      const auto &fp2 = matches[i];
      pixel_dist +=
          static_cast<double>((fp1->imagePoint() - fp2->imagePoint()).norm());
    }
  }

  pixel_dist /= double(query_frame_points.size());

  if (pixel_dist < 4)
    return {};

  // Get map points
  std::vector<std::shared_ptr<MapPoint>> map_points(query_frame_points.size(),
                                                    nullptr);

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

  // Add to index
  Eigen::VectorXf embedding = key_frame->imageEmbedding();
  std::vector<float> vec(embedding.data(), embedding.data() + embedding.size());
  index_.add(1, vec.data());
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

std::vector<std::shared_ptr<KeyFrame>> Map::keyFrames() {
  std::vector<std::shared_ptr<KeyFrame>> key_frames;
  key_frames.reserve(key_frames_.size());
  for (auto it = key_frames_.begin(); it != key_frames_.end();
       ++it) {
    key_frames.push_back(it->second);
  }
  return key_frames;
}

std::size_t Map::numMapPoints() const { return map_points_.size(); }

std::vector<std::shared_ptr<MapPoint>> Map::mapPoints() {
  std::vector<std::shared_ptr<MapPoint>> map_points;
  for (auto it = map_points_.begin(); it != map_points_.end();
       ++it) {
    map_points.push_back(it->second);
  }
  return map_points;
}

double Map::evaluateError() const {
  double error{0};
  std::for_each(
      key_frames_.begin(), key_frames_.end(),
      [&](const std::pair<std::size_t, std::shared_ptr<KeyFrame>> &kf_pair) {
        error += kf_pair.second->evaluateError();
      });
  return error;
}

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

  std::vector<float> query(embedding.data(),
                           embedding.data() + embedding.size());

  // Query from keyframe
  faiss::idx_t *I = new faiss::idx_t[static_cast<std::size_t>(k)];
  float *D = new float[static_cast<std::size_t>(k)];
  index_.search(1, query.data(), k, D, I);

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