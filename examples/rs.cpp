#include <iostream>
#include <filesystem>

#include <librealsense2/rs.hpp>

#include "taey/TAEY.h"

int main(int argc, char **argv) {
    rs2::pipeline pipe;

    // 2. Configure the pipeline (optional, starts with default settings if omitted)
    rs2::config cfg;
    // Example config: enable depth stream at 640x480 resolution, 30 fps
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    // Example config: enable color stream
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);
    pipe.start(cfg);

    YAML::Node config;
    TAEY taey(argc, argv, config);
    std::thread thread([&]() {
        while (true) {
            // Block program until frames arrive
            rs2::frameset frames = align_to_color.process(pipe.wait_for_frames());

            // Try to get a frame of a depth image
            auto depth = frames.get_depth_frame();
            auto color = frames.get_color_frame();

            const int w = color.get_width();
            const int h = color.get_height();

            cv::Mat depth_cv(cv::Size(w, h), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat color_cv(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
            
            std::shared_ptr<KeyFrame> key_frame = taey(color_cv, depth_cv);
        }
    });

    int ret = taey.exec();

    if (thread.joinable())
        thread.join();

   return ret;
}