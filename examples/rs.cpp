#include <iostream>
#include <filesystem>

#include <librealsense2/rs.hpp>
#include <opencv2/rgbd.hpp>

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

    int margin_w = int(float(intrinsics.width) * 0.08);
    int margin_h = int(float(intrinsics.height) * 0.08);

    YAML::Node config;
    config["encoder"] = std::string{"models/clip/clip.engine"};

    // 2. Set Config with UPDATED Intrinsics
    config["fx"] = intrinsics.fx;
    config["fy"] = intrinsics.fy;
    
    // Principal point shifts because the (0,0) coordinate moved
    config["cx"] = intrinsics.ppx - static_cast<float>(margin_w); 
    config["cy"] = intrinsics.ppy - static_cast<float>(margin_h);
    
    // Resolution decreases by 2x margin (left+right, top+bottom)
    config["width"] = intrinsics.width - (2 * margin_w); 
    config["height"] = intrinsics.height - (2 * margin_h);
    
    config["depth_scale"] = 1.0;
    config["distortion"] = std::vector<float>(std::begin(intrinsics.coeffs), std::end(intrinsics.coeffs));
    cv::Rect roi(margin_w, margin_h, config["width"].as<int>(), config["height"].as<int>());
    
    TAEY taey(argc, argv, config);
    cv::rgbd::DepthCleaner* depthc = new cv::rgbd::DepthCleaner(CV_16U, 7, cv::rgbd::DepthCleaner::DEPTH_CLEANER_NIL);

    std::thread thread([&]() {

        rs2::spatial_filter spatial;
        spatial.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
        spatial.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5);
        spatial.set_option(RS2_OPTION_HOLES_FILL, 0);

        rs2::temporal_filter temporal;
        temporal.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f); // Smoothness
        temporal.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20); // Threshold

        rs2::hole_filling_filter hole_filter;
        hole_filter.set_option(RS2_OPTION_HOLES_FILL, 0);

        while (true) {
            // Block program until frames arrive
            frames = align_to_color.process(pipe.wait_for_frames());

            // Try to get a frame of a depth image
            auto depth = frames.get_depth_frame();
            auto color = frames.get_color_frame();

            depth = spatial.process(depth);
            depth = temporal.process(depth);
            depth = hole_filter.process(depth);

            const int w = color.get_width();
            const int h = color.get_height();

            cv::Mat depth_raw(cv::Size(w, h), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat depth_clean(cv::Size(w, h), CV_16U);
    
            depthc->operator()(depth_raw, depth_clean);

            depth_raw.copyTo(depth_clean); 
            
            cv::Mat depth_f;
            depth_clean.convertTo(depth_f, CV_32F, aligned_depth_frame.get_units());

            cv::Mat color_img(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

            // Crop both images
            cv::Mat depth_cropped = depth_f(roi).clone();
            cv::Mat color_cropped = color_img(roi).clone();
            
            std::shared_ptr<KeyFrame> key_frame = taey(color_cropped, depth_cropped);
        }
    });

    int ret = taey.exec();

    if (thread.joinable())
        thread.join();

   return ret;
}