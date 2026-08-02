#include <memory>
#include <thread>
#include <vector>

#include <librealsense2/rs.hpp>

#include "taey/Config.h"
#include "taey/KeyFrame.h"
#include "taey/TAEY.h"
#include "taey/Logger.h"

int main(int argc, char **argv) {
    YAML::Node config = taey::loadConfig("config.yaml");

    rs2::config cfg;
    const int rs_w = config["rs_width"].as<int>();
    const int rs_h = config["rs_height"].as<int>();
    const int rs_fps = config["rs_fps"].as<int>();
    cfg.enable_stream(RS2_STREAM_DEPTH, rs_w, rs_h, RS2_FORMAT_Z16, rs_fps);
    cfg.enable_stream(RS2_STREAM_COLOR, rs_w, rs_h, RS2_FORMAT_BGR8, rs_fps);

    rs2::pipeline pipe;
    rs2::align align_to_color(RS2_STREAM_COLOR);
    pipe.start(cfg);

    rs2::frameset frames = align_to_color.process(pipe.wait_for_frames());
    rs2::depth_frame depth_frame = frames.get_depth_frame();
    const float depth_units = depth_frame.get_units();
    rs2_intrinsics intrinsics =
        depth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();

    // Crop a margin from each edge and shift the camera model accordingly.
    const float rs_margin = config["rs_margin"].as<float>();
    const int margin_w = int(float(intrinsics.width) * rs_margin);
    const int margin_h = int(float(intrinsics.height) * rs_margin);
    config["fx"] = intrinsics.fx;
    config["fy"] = intrinsics.fy;
    config["cx"] = intrinsics.ppx - static_cast<float>(margin_w);
    config["cy"] = intrinsics.ppy - static_cast<float>(margin_h);
    config["width"] = intrinsics.width - (2 * margin_w);
    config["height"] = intrinsics.height - (2 * margin_h);
    config["depth_scale"] = 1.0;
    config["distortion"] = std::vector<float>(std::begin(intrinsics.coeffs),
                                              std::end(intrinsics.coeffs));
    cv::Rect roi(margin_w, margin_h, config["width"].as<int>(), config["height"].as<int>());

    // Declared before the `taey` variable shadows the taey:: namespace.
    taey::Logger logger("rs");
    TAEY taey(config);

    std::thread thread([&]() {
        rs2::spatial_filter spatial;
        spatial.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
        spatial.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5);
        spatial.set_option(RS2_OPTION_HOLES_FILL, 0);

        rs2::temporal_filter temporal;
        temporal.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
        temporal.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20);

        rs2::hole_filling_filter hole_filter;
        hole_filter.set_option(RS2_OPTION_HOLES_FILL, 0);

        while (true) {
            rs2::frameset frames = align_to_color.process(pipe.wait_for_frames());
            auto depth = frames.get_depth_frame();
            auto color = frames.get_color_frame();

            depth = spatial.process(depth);
            depth = temporal.process(depth);
            depth = hole_filter.process(depth);

            const int w = color.get_width();
            const int h = color.get_height();
            cv::Mat depth_raw(cv::Size(w, h), CV_16U, (void*)depth.get_data(),
                              cv::Mat::AUTO_STEP);
            cv::Mat color_img(cv::Size(w, h), CV_8UC3, (void*)color.get_data(),
                              cv::Mat::AUTO_STEP);

            cv::Mat depth_f;
            depth_raw.convertTo(depth_f, CV_32F, depth_units);

            cv::Mat depth_cropped = depth_f(roi).clone();
            cv::Mat color_cropped = color_img(roi).clone();

            logger.logImage(color_cropped, depth_cropped);

            // Sensor timestamp, milliseconds to seconds.
            double timestamp = color.get_timestamp() / 1000.0;
            std::shared_ptr<KeyFrame> key_frame =
                taey(color_cropped, depth_cropped, timestamp);
            if (key_frame != nullptr) {
                logger.logPoints(key_frame);
            }
        }
    });

    if (thread.joinable())
        thread.join();
}
