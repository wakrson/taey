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

// Attach a Rerun recording stream to a sink chosen by config. The matching
// environment variable, if set, overrides the config value for one-off runs.
// Resolution order, highest priority first:
//   RERUN_SAVE    / rerun_save:    record to an .rrd file (headless)
//   RERUN_ADDRESS / rerun_address: connect to a running viewer over gRPC
//   (none set)                     spawn a local native viewer
inline void connectRerun(rerun::RecordingStream &rec, const YAML::Node &config) {
  auto resolve = [&](const char *env, const char *key) -> std::string {
    if (const char *value = std::getenv(env)) return value;
    if (config[key] && !config[key].IsNull()) return config[key].as<std::string>();
    return {};
  };

  const std::string save = resolve("RERUN_SAVE", "rerun_save");
  const std::string address = resolve("RERUN_ADDRESS", "rerun_address");

  if (!save.empty()) {
    rec.save(save).exit_on_failure();
  } else if (!address.empty()) {
    rec.connect_grpc(address).exit_on_failure();
  } else {
    rec.spawn().exit_on_failure();
  }
}

}  // namespace taey
