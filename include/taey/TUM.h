#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

class TUM {
public:
  TUM(const std::filesystem::path &);
  cv::Mat getImage(const double &) const;
  cv::Mat getDepth(const double &) const;
  std::size_t getId(const double &) const;
  Eigen::Transform<double, 3, Eigen::Isometry>
  getGroundtruth(const double &) const;
  std::map<double, std::string> parse(const std::filesystem::path &);
  Eigen::Transform<double, 3, Eigen::Isometry> getPose(const double &);
  std::vector<double> timestamps() const;
  double getNearestKey(const double &,
                       const std::map<double, std::string> &) const;

private:
  std::vector<double> timestamps_;
  std::filesystem::path base_dir_;
  std::map<double, std::string> image_map_, depth_map_, groundtruth_map_;
  std::vector<double> loadTimestamps(const std::filesystem::path &);
  YAML::Node config_;
};