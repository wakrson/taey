#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

#include <yaml-cpp/yaml.h>
#include <rerun.hpp>

namespace taey {

// Load the repo-level config.yaml, then overlay a dataset's calibration.yaml
// on top so dataset-specific values (intrinsics, distortion, resolution,
// depth_scale) win over the application defaults. Either path may be empty or
// missing; whatever exists is merged.
inline YAML::Node loadConfig(const std::filesystem::path &base,
                             const std::filesystem::path &overlay = {}) {
  YAML::Node config;
  if (!base.empty() && std::filesystem::exists(base)) {
    config = YAML::LoadFile(base.string());
  }
  if (!overlay.empty() && std::filesystem::exists(overlay)) {
    YAML::Node node = YAML::LoadFile(overlay.string());
    for (const auto &kv : node) {
      config[kv.first.as<std::string>()] = kv.second;
    }
  }
  return config;
}

}  // namespace taey
