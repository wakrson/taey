#include <iostream>
#include <filesystem>

#include <librealsense2/rs.hpp>
#include <opencv2/rgbd.hpp>

#include <rerun.hpp>

#include "taey/KeyFrame.h"
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

    // Rerun recording stream. Three sinks, in priority order:
    //   RERUN_SAVE=<path>   record to an .rrd file — headless.
    //   RERUN_ADDRESS=<a>   connect to an already-running viewer over gRPC.
    //   (neither)           spawn a local native viewer — interactive default.
    rerun::RecordingStream rec("taey/rs");
    if (const char *path = std::getenv("RERUN_SAVE")) {
        rec.save(path).exit_on_failure();
    } else if (const char *addr = std::getenv("RERUN_ADDRESS")) {
        rec.connect_grpc(addr).exit_on_failure();
    } else {
        rec.spawn().exit_on_failure();
    }

    // Log a tracked keyframe: RGB, depth, and its world-frame point cloud.
    auto log_key_frame = [&rec](const std::shared_ptr<KeyFrame> &kf) {
        rec.set_time_sequence("keyframe", static_cast<int64_t>(kf->id()));

        cv::Mat rgb = kf->image();
        if (!rgb.empty()) {
            cv::Mat out;
            cv::cvtColor(rgb, out,
                         rgb.channels() == 3 ? cv::COLOR_BGR2RGB
                                             : cv::COLOR_GRAY2RGB);
            rec.log("camera/rgb",
                    rerun::Image::from_rgb24(
                        rerun::Collection<uint8_t>::borrow(out.data,
                                                           out.total() * 3),
                        {static_cast<uint32_t>(out.cols),
                         static_cast<uint32_t>(out.rows)}));
        }

        cv::Mat depth = kf->depth();
        if (!depth.empty()) {
            cv::Mat depth_f;
            if (depth.type() != CV_32F) {
                depth.convertTo(depth_f, CV_32F);
            } else {
                depth_f = depth.isContinuous() ? depth : depth.clone();
            }
            rec.log("camera/depth",
                    rerun::DepthImage(
                        rerun::Collection<float>::borrow(
                            reinterpret_cast<const float *>(depth_f.data),
                            depth_f.total()),
                        {static_cast<uint32_t>(depth_f.cols),
                         static_cast<uint32_t>(depth_f.rows)}));
        }

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
            if (key_frame != nullptr) {
                log_key_frame(key_frame);
            }
        }
    });

    int ret = taey.exec();

    if (thread.joinable())
        thread.join();

   return ret;
}