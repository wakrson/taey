#include <fstream>
#include <iostream>
#include <string>

#include "TUM.h"

TUM::TUM(const std::filesystem::path &path) {
  this->base_dir_ = path;
  this->timestamps_ =
      this->loadTimestamps(base_dir_ / std::filesystem::path("rgb.txt"));
  this->image_map_ = this->parse(base_dir_ / std::filesystem::path("rgb.txt"));
  this->depth_map_ =
      this->parse(base_dir_ / std::filesystem::path("depth.txt"));
  this->groundtruth_map_ =
      this->parse(base_dir_ / std::filesystem::path("groundtruth.txt"));
}

double TUM::getNearestKey(const double &key,
                          const std::map<double, std::string> &mp) const {
  auto it = mp.lower_bound(key);
  if (it == mp.end()) {
    return std::prev(it)->first;
  }

  if (std::abs(key - it->first) <= std::abs(key - std::prev(it)->first)) {
    return it->first;
  }

  return std::prev(it)->first;
}

cv::Mat TUM::getImage(const double &timestamp) const {
  double key = getNearestKey(timestamp, this->image_map_);
  cv::Mat image = cv::imread(this->image_map_.at(key));
  return image;
}

cv::Mat TUM::getDepth(const double &timestamp) const {
  double key = getNearestKey(timestamp, this->depth_map_);
  cv::Mat depth_raw =
      cv::imread(this->depth_map_.at(key), cv::IMREAD_UNCHANGED);
  cv::Mat depth;
  depth_raw.convertTo(depth, CV_32F);
  return depth;
}

std::size_t TUM::getId(const double &timestamp) const {
  auto it = std::find(timestamps_.begin(), timestamps_.end(), timestamp);
  return static_cast<std::size_t>(std::distance(timestamps_.begin(), it));
}

Eigen::Transform<double, 3, Eigen::Isometry>
TUM::getGroundtruth(const double &timestamp) const {
  double key = getNearestKey(timestamp, this->groundtruth_map_);
  std::string pose_str = this->groundtruth_map_.at(key);
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
    if (line[0] != '#') {
      std::istringstream iss(line);
      std::string timestamp, data;
      std::getline(iss, timestamp, ' ');
      std::getline(iss, data);
      if (path == std::filesystem::path("groundtruth.txt")) {
        data_map[std::stod(timestamp)] = data;
      } else {
        data_map[std::stod(timestamp)] =
            this->base_dir_ / std::filesystem::path(data);
      }
    }
  }
  file.close();
  return data_map;
}

std::vector<double> TUM::timestamps() const { return this->timestamps_; }

std::vector<double> TUM::loadTimestamps(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Failed to open : " << std::filesystem::absolute(path)
              << std::endl;
  }

  std::string line;
  std::vector<double> timestamps;
  while (std::getline(file, line)) {
    if (line[0] != '#') {
      std::istringstream iss(line);
      std::string timestamp;
      iss >> timestamp;
      timestamps.push_back(std::stod(timestamp));
    }
  }
  file.close();
  return timestamps;
}
