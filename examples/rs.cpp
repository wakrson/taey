#include <iostream>
#include <filesystem>

#include <librealsense2/rs.hpp>

#include "taey/TAEY.h"

int main(int argc, char **argv) {
    rs2::pipeline pipe;

    // Configure the pipeline (optional, starts with default settings if omitted)
    rs2::config cfg;
    // Example config: enable depth stream at 640x480 resolution, 30 fps
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    // Example config: enable color stream
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);

    rs2::pipeline_profile profile = pipe.start(cfg);
    rs2::frameset frames = align_to_color.process(pipe.wait_for_frames());
    rs2::depth_frame aligned_depth_frame = frames.get_depth_frame();
    rs2_intrinsics intrinsics = aligned_depth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();

    YAML::Node config;
    config["encoder"] = std::string{"models/clip.engine"};
    config["fx"] = intrinsics.fx;
    config["fy"] = intrinsics.fy;
    config["cx"] = intrinsics.ppx;
    config["cy"] = intrinsics.ppy;
    config["width"] = intrinsics.width;
    config["height"] = intrinsics.height;
    config["depth_scale"] = 1.0;
    config["distortion"] = std::vector<float>(std::begin(intrinsics.coeffs), std::end(intrinsics.coeffs));

    TAEY taey(argc, argv, config);
    std::thread thread([&]() {
        while (true) {
            // Block program until frames arrive
            frames = align_to_color.process(pipe.wait_for_frames());

            // Try to get a frame of a depth image
            auto depth = frames.get_depth_frame();
            auto color = frames.get_color_frame();

            const int w = color.get_width();
            const int h = color.get_height();

            cv::Mat depth_cv(cv::Size(w, h), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat color_cv(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

            depth_cv.convertTo(depth_cv, CV_32F, aligned_depth_frame.get_units());
            std::shared_ptr<KeyFrame> key_frame = taey(color_cv, depth_cv);
        }
    });

    int ret = taey.exec();

    if (thread.joinable())
        thread.join();

   return ret;
}