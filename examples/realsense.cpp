#include <iostream>
#include <filesystem>

#include "taey/TUM.h"
#include "taey/TAEY.h"

int main() {
    /*
    TAEY taey;
    // Create a Pipeline - this serves as a top-level API for streaming and processing frames
    rs2::pipeline p;

    // Configure and start the pipeline
    p.start();

    rs2::context ctx( settings.dump() );
    std::string serial;
    if( ! device_with_streams( ctx, { RS2_STREAM_COLOR, RS2_STREAM_DEPTH }, serial ) )
        return EXIT_SUCCESS;

    // Create and initialize GUI related objects
    window app(1280, 720, "RealSense Align Example"); // Simple window handling
    // Setup Dear ImGui context
    ImGui::CreateContext();
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(app, false);
    ImGui_ImplOpenGL3_Init();
    rs2::colorizer c;                     // Helper to colorize depth images
    texture depth_image, color_image;     // Helpers for renderig images

    // Create a pipeline to easily configure and start the camera
    rs2::pipeline pipe( ctx );
    rs2::config cfg;
    if (!serial.empty())
        cfg.enable_device(serial);
    cfg.enable_stream(RS2_STREAM_DEPTH);
    cfg.enable_stream(RS2_STREAM_COLOR);
    pipe.start(cfg);

    // Define two align objects. One will be used to align
    // to depth viewport and the other to color.
    // Creating align object is an expensive operation
    // that should not be performed in the main loop
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);
    while (true) {
        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames();

        // Try to get a frame of a depth image
        rs2::depth_frame depth = frames.get_depth_frame();

        // Get the depth frame's dimensions
        float width = depth.get_width();
        float height = depth.get_height();

        std::shared_ptr<KeyFrame> key_frame = taey(timestamp, image, depth);
    }
    */
   return 0;
}