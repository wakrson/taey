#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "taey/Config.h"
#include "taey/TUM.h"
#include "taey/KeyFrame.h"
#include "taey/Map.h"
#include "taey/TAEY.h"
#include "taey/Logger.h"

int main(int argc, char** argv) {
    std::filesystem::path dataset_path("datasets/rgbd_dataset_freiburg3_nostructure_texture_near_withloop");

    if (argc > 1) {
        dataset_path = std::filesystem::path(argv[1]);
    }

    if (!std::filesystem::exists(dataset_path)) {
        std::cerr << "Dataset not found: " << dataset_path << std::endl;
        return 1;
    }

    // Optional second argument overrides the results root (see bash/eval_tum.sh).
    const std::filesystem::path results_root(argc > 2 ? argv[2] : "results");
    const std::filesystem::path results_dir = results_root / dataset_path.filename();
    std::filesystem::create_directories(results_dir);
    const std::filesystem::path output_path = results_dir / (dataset_path.filename().string() + ".txt");

    YAML::Node config = taey::loadConfig("config.yaml", dataset_path / "calibration.yaml");

    TUM tum(dataset_path);
    taey::Logger rec("tum");
    TAEY taey(config);

    // Log a tracked keyframe: RGB, depth, and its world-frame point cloud.

    std::thread thread([&]() {
        // Trajectory is collected in memory and written once at the end.
        std::vector<std::string> trajectory;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Single pass; keyframe selection happens inside taey().
        for (const double& timestamp : tum.timestamps()) {
            cv::Mat image = tum.getImage(timestamp);
            cv::Mat depth = tum.getDepth(timestamp);
            // Skip frames with no valid RGB / depth association.
            if (image.empty() || depth.empty()) {
                continue;
            }
            rec.logImage(image, depth);
            std::shared_ptr<KeyFrame> key_frame = taey(image, depth, timestamp);
            if (key_frame != nullptr) {
                if (taey.loopClosed()) {
                    // Re-log every keyframe so the viewer reflects the
                    // loop-optimized poses.
                    const std::shared_ptr<Map> map = taey.map();
                    for (std::size_t i = 0; i < map->numKeyFrames(); i++) {
                        rec.logPoints(map->keyFrame(i));
                    }
                } else {
                    rec.logPoints(key_frame);
                }
                Eigen::Isometry3d pose = key_frame->pose();
                Eigen::Vector3d translation = pose.translation();
                Eigen::Quaterniond quat(pose.rotation());
                // timestamp tx ty tz qx qy qz qw
                std::ostringstream line;
                line << std::to_string(timestamp) << " "
                     << translation(0) << " " << translation(1) << " "
                     << translation(2) << " " << quat.x() << " "
                     << quat.y() << " " << quat.z() << " " << quat.w();
                trajectory.push_back(line.str());
            }
        }

        std::ofstream output_file(output_path);
        if (!output_file.is_open()) {
            std::cerr << "Failed to open output file: " << output_path << std::endl;
            return;
        }
        for (const std::string& line : trajectory) {
            output_file << line << std::endl;
        }
        std::cout << "Wrote trajectory: " << output_path << std::endl;
    });

    if (thread.joinable()) {
        thread.join();
    }
}
