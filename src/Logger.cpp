#include <cstdlib>
#include <iostream>

#include <opencv2/imgproc.hpp>

#include "KeyFrame.h"
#include "Logger.h"

taey::Logger::Logger(std::string app_id)
    : rec(std::move(app_id))
{
    if (!rec.is_enabled()) {
        return;
    }
    auto err = rec.spawn();   // launch the native Rerun viewer and connect
    if (!err.is_ok()) {
        std::cerr << "Failed to spawn Rerun viewer: "
                  << err.description << std::endl;
        std::exit(1);
    }
    std::cout << "Rerun native viewer launched." << std::endl;
}

void taey::Logger::logImage(const cv::Mat& rgb, const cv::Mat& depth) {
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

    if (!depth.empty()) {
        cv::Mat depth_f = depth.clone();
        if (depth.type() != CV_32F) {
            depth_f.convertTo(depth_f, CV_32F);
        }
        rec.log(
            "camera/depth",
            rerun::DepthImage(
                reinterpret_cast<const float*>(depth_f.data),
                {static_cast<uint32_t>(depth_f.cols), static_cast<uint32_t>(depth_f.rows)})
            );
    }
}

void taey::Logger::logPoints(const std::shared_ptr<KeyFrame>& key_frame) {
    // Skip the dense cloud generation entirely when running headless.
    if (!rec.is_enabled()) {
        return;
    }
    // Skip re-logs whose pose moved less than 1 cm / 0.01 rad; the logged
    // cloud is still accurate at that scale.
    const Eigen::Isometry3d pose = key_frame->pose();
    const auto it = logged_poses_.find(key_frame->id());
    if (it != logged_poses_.end()) {
        const Eigen::Isometry3d delta = it->second.inverse() * pose;
        if (delta.translation().norm() < 0.01 &&
            Eigen::AngleAxisd(delta.rotation()).angle() < 0.01) {
            return;
        }
    }
    logged_poses_[key_frame->id()] = pose;
    Eigen::MatrixXd pW;
    key_frame->objectPoints(pW);
    // Subsample the dense per-pixel cloud so the viewer stays responsive as
    // keyframes accumulate (visualization only; does not affect the map).
    const Eigen::Index point_stride = 10;
    std::vector<rerun::Position3D> positions;
    std::vector<rerun::Color> colors;
    positions.reserve(static_cast<std::size_t>(pW.rows() / point_stride + 1));
    colors.reserve(static_cast<std::size_t>(pW.rows() / point_stride + 1));
    for (Eigen::Index i = 0; i < pW.rows(); i += point_stride) {
        const auto pt = pW.row(i);
        positions.emplace_back(pt(0), pt(1), pt(2));
        // cloud stores [X, Y, Z, B, G, R] -> Color(r, g, b)
        colors.emplace_back(pt(5), pt(4), pt(3));
    }
    rec.log("map/points/" + std::to_string(key_frame->id()), rerun::Points3D(positions).with_colors(colors).with_radii(0.01f));
}