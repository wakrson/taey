#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "TUM.h"

// RGB<->depth association window (TUM's associate.py uses 0.02 s).
const double max_assoc_dt = 0.02; // seconds
// Announce recording holes larger than this at load time.
const double gap_warn_dt = 0.5; // seconds

TUM::TUM(const std::filesystem::path &path) {
  base_dir_ = path;
  image_map_ = parse(base_dir_ / std::filesystem::path("rgb.txt"));
  depth_map_ = parse(base_dir_ / std::filesystem::path("depth.txt"));
  groundtruth_map_ = parse(base_dir_ / std::filesystem::path("groundtruth.txt"));
  if (image_map_.empty() || depth_map_.empty()) {
    throw std::runtime_error("TUM: no RGB/depth data in " + path.string());
  }

  // Expose only frames with a valid RGB<->depth association.
  std::size_t unassociated = 0;
  for (const auto &kv : image_map_) {
    if (std::abs(kv.first - getNearestKey(kv.first, depth_map_)) <= max_assoc_dt) {
      timestamps_.push_back(kv.first);
    } else {
      unassociated++;
    }
  }
  if (unassociated > 0) {
    std::cerr << "TUM: dropped " << unassociated
              << " frames with no depth within " << max_assoc_dt << " s"
              << std::endl;
  }

  // Announce recording holes (e.g. fr2_pioneer_slam2 misses two ~3.3 s spans).
  for (std::size_t i = 1; i < timestamps_.size(); i++) {
    const double dt = timestamps_[i] - timestamps_[i - 1];
    if (dt > gap_warn_dt) {
      std::cerr << "TUM: " << dt << " s recording gap at t=" << std::fixed
                << timestamps_[i - 1] << std::defaultfloat << std::endl;
    }
  }
}

double TUM::getNearestKey(const double &key,
                          const std::map<double, std::string> &mp) const {
  if (mp.empty()) {
    return key;
  }
  auto it = mp.lower_bound(key);
  if (it == mp.end()) {
    return std::prev(it)->first;
  }
  // No earlier element exists; don't step before begin().
  if (it == mp.begin()) {
    return it->first;
  }

  if (std::abs(key - it->first) <= std::abs(key - std::prev(it)->first)) {
    return it->first;
  }

  return std::prev(it)->first;
}

cv::Mat TUM::getImage(const double &timestamp) const {
  double key = getNearestKey(timestamp, image_map_);
  // Never serve an image far from the requested time.
  if (std::abs(timestamp - key) > max_assoc_dt) {
    return cv::Mat();
  }
  cv::Mat image = cv::imread(image_map_.at(key));
  return image;
}

cv::Mat TUM::getDepth(const double &timestamp) const {
  double key = getNearestKey(timestamp, depth_map_);
  if (std::abs(timestamp - key) > max_assoc_dt) {
    return cv::Mat();
  }
  cv::Mat depth_raw = cv::imread(depth_map_.at(key), cv::IMREAD_UNCHANGED);
  cv::Mat depth_f;
  // Keep raw sensor units; Camera applies depth_scale during back-projection.
  depth_raw.convertTo(depth_f, CV_32F);
  return depth_f;
}

std::size_t TUM::getId(const double &timestamp) const {
  auto it = std::find(timestamps_.begin(), timestamps_.end(), timestamp);
  return static_cast<std::size_t>(std::distance(timestamps_.begin(), it));
}

Eigen::Transform<double, 3, Eigen::Isometry>
TUM::getGroundtruth(const double &timestamp) const {
  double key = getNearestKey(timestamp, groundtruth_map_);
  std::string pose_str = groundtruth_map_.at(key);
  std::istringstream iss(pose_str);

  double tx, ty, tz, qx, qy, qz, qw;
  iss >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

  Eigen::Quaterniond q(qw, qx, qy, qz);
  q.normalize();

  Eigen::Isometry3d groundtruth = Eigen::Isometry3d::Identity();
  groundtruth.linear() = q.toRotationMatrix();
  groundtruth.translation() << tx, ty, tz;

  return groundtruth;
}

std::map<double, std::string> TUM::parse(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Failed to open : " << std::filesystem::absolute(path)
              << std::endl;
  }

  std::string line;
  std::map<double, std::string> data_map;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    std::string timestamp, data;
    std::getline(iss, timestamp, ' ');
    std::getline(iss, data);
    if (path.filename() == "groundtruth.txt") {
      data_map[std::stod(timestamp)] = data;
    } else {
      data_map[std::stod(timestamp)] =
          base_dir_ / std::filesystem::path(data);
    }
  }
  file.close();
  return data_map;
}

std::vector<double> TUM::timestamps() const { return timestamps_; }
