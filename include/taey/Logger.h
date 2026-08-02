#pragma once

#include <map>
#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <rerun.hpp>

class KeyFrame;

namespace taey {

// Named taey::Logger to avoid colliding with tensorrt-cpp-api's Logger.
class Logger {
private:
    rerun::RecordingStream rec;
    // Pose each keyframe's cloud was last logged at; re-logs are skipped
    // while the pose is unchanged.
    std::map<std::size_t, Eigen::Isometry3d> logged_poses_;
public:
    Logger(std::string app_id = "taey");
    void logImage(const cv::Mat& rgb, const cv::Mat& depth);
    void logPoints(const std::shared_ptr<KeyFrame>& key_frame);
};

} // namespace taey
