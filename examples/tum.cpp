#include <iostream>
#include <filesystem>

#include "taey/TUM.h"
#include "taey/KeyFrame.h"
#include "taey/TAEY.h"

int main(int argc, char** argv) {
    std::filesystem::path dataset_path("datasets/rgbd_dataset_freiburg2_pioneer_slam2");
    TUM tum(dataset_path);
    YAML::Node config = YAML::LoadFile(dataset_path / "calibration.yaml");

    TAEY taey(argc, argv, config);
    std::thread thread([&]() {
        int n = 10;  // process every n-th frame
        int frame_idx = 0;
        std::ofstream output_file(dataset_path / "output.txt");
        if (output_file.is_open()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            while (true) {
                for (const double& timestamp : tum.timestamps()) {
                    if (frame_idx % n == 0) {
                        cv::Mat image = tum.getImage(timestamp);
                        cv::Mat depth = tum.getDepth(timestamp);
                        std::shared_ptr<KeyFrame> key_frame = taey(image, depth);
                        if (key_frame != nullptr) {
                            // Write output to file
                            Eigen::Isometry3d pose = key_frame->pose();
                            Eigen::Vector3d translation = pose.translation();
                            Eigen::Quaterniond quat (pose.rotation());
                            // Write timestamp
                            output_file << std::to_string(timestamp) << " ";
                            // Write translation
                            output_file << translation(0) << " " << translation(1) << " " << translation(2) << " ";
                            // Write rotation
                            output_file << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
                            // Sleep
                            std::this_thread::sleep_for(std::chrono::milliseconds(33));
                        }
                    }
                    frame_idx++;
                }
                taey.reset();
            }
        }
    });

    int ret = taey.exec();

    if (thread.joinable()) {
        thread.join();
    }

    return ret;
}