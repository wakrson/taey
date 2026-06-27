#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <QMetaObject>

#include <rerun.hpp>

#include "taey/Config.h"
#include "taey/TUM.h"
#include "taey/KeyFrame.h"
#include "taey/TAEY.h"

int main(int argc, char** argv) {
    std::filesystem::path dataset_path("datasets/rgbd_dataset_freiburg2_pioneer_slam2");

    if (argc > 1) {
        dataset_path = std::filesystem::path(argv[1]);
    }

    if (!std::filesystem::exists(dataset_path)) {
        std::cerr << "Dataset not found: " << dataset_path << std::endl;
        return 1;
    }

    // Estimated trajectory (TUM format) is written to results/<scene>.txt so a
    // batch runner can hand it straight to evo.
    const std::filesystem::path results_dir("results");
    std::filesystem::create_directories(results_dir);
    const std::filesystem::path output_path = results_dir / (dataset_path.filename().string() + ".txt");

    YAML::Node config = taey::loadConfig("config.yaml", dataset_path / "calibration.yaml");

    TUM tum(dataset_path);
    TAEY taey(argc, argv, config);

    rerun::RecordingStream rec("taey/tum");
    rec.spawn().exit_on_failure();

    // Log a tracked keyframe: RGB, depth, and its world-frame point cloud.
    auto log_key_frame = [&rec](const std::shared_ptr<KeyFrame> &kf) {
        rec.set_time_sequence("keyframe", static_cast<int64_t>(kf->id()));

        cv::Mat rgb = kf->image();
        if (!rgb.empty()) {
            cv::Mat out;
            if (rgb.channels() == 3)
                cv::cvtColor(rgb, out, cv::COLOR_BGR2RGB);
            else
                cvtColor(rgb, out, cv::COLOR_GRAY2RGB);
            
            rec.log(
                "camera/rgb",rerun::Image::from_rgb24(
                    rerun::Collection<uint8_t>::borrow(out.data, out.total() * 3),
                    {static_cast<uint32_t>(out.cols), static_cast<uint32_t>(out.rows)})
                );
        }

        cv::Mat depth = kf->depth();
        if (!depth.empty()) {
            cv::Mat depth_f = depth.clone();
            if (depth.type() != CV_32F) {
                depth_f.convertTo(depth_f, CV_32F);
            }
            rec.log(
                "camera/depth",
                rerun::DepthImage(
                    depth_f.data,
                    {static_cast<uint32_t>(depth_f.cols), static_cast<uint32_t>(depth_f.rows)})
                );
        }

        // Per-keyframe world points. Logging each keyframe under its own entity
        // path lets them accumulate in the viewer without a global map buffer.
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        kf->objectPoints(cloud);
        std::vector<rerun::Position3D> positions;
        std::vector<rerun::Color> colors;
        positions.reserve(cloud->size());
        colors.reserve(cloud->size());
        for (const auto &pt : cloud->points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
                !std::isfinite(pt.z)) {
                continue;
            }
            if (pt.x == 0.0f && pt.y == 0.0f && pt.z == 0.0f) continue;
            positions.emplace_back(pt.x, pt.y, pt.z);
            colors.emplace_back(pt.r, pt.g, pt.b);
        }
        rec.log("map/points/" + std::to_string(kf->id()),
                rerun::Points3D(positions).with_colors(colors).with_radii(0.01f));
    };

    std::thread thread([&]() {
        const int n = config["stride"].as<int>();  // process every n-th frame
        int frame_idx = 0;

        std::ofstream output_file(output_path);
        if (!output_file.is_open()) {
            std::cerr << "Failed to open output file: " << output_path << std::endl;
            QMetaObject::invokeMethod(&taey, "quit", Qt::QueuedConnection);
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Single pass over the sequence (no infinite loop) so the run
        // terminates on its own and can be batched over many scenes.
        for (const double& timestamp : tum.timestamps()) {
            if (frame_idx % n == 0) {
                cv::Mat image = tum.getImage(timestamp);
                cv::Mat depth = tum.getDepth(timestamp);
                std::shared_ptr<KeyFrame> key_frame = taey(image, depth);
                if (key_frame != nullptr) {
                    log_key_frame(key_frame);
                    Eigen::Isometry3d pose = key_frame->pose();
                    Eigen::Vector3d translation = pose.translation();
                    Eigen::Quaterniond quat(pose.rotation());
                    // timestamp tx ty tz qx qy qz qw
                    output_file << std::to_string(timestamp) << " "
                                << translation(0) << " " << translation(1) << " "
                                << translation(2) << " " << quat.x() << " "
                                << quat.y() << " " << quat.z() << " " << quat.w()
                                << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(33));
                }
            }
            frame_idx++;
        }

        output_file.close();
        std::cout << "Wrote trajectory: " << output_path << std::endl;

        // Quit the Qt event loop so exec() returns and the process exits.
        QMetaObject::invokeMethod(&taey, "quit", Qt::QueuedConnection);
    });

    int ret = taey.exec();

    if (thread.joinable()) {
        thread.join();
    }

    return ret;
}
