#include <iostream>
#include <filesystem>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include <rerun.hpp>

#include "taey/Config.h"
#include "taey/TUM.h"
#include "taey/CLIP.h"
#include "taey/KeyFrame.h"

int main(int argc, char** argv) {
    std::filesystem::path dataset_path("datasets/rgbd_dataset_freiburg2_pioneer_slam2");

    if (argc > 1) {
        dataset_path = std::filesystem::path(argv[1]);
    }

    YAML::Node config = taey::loadConfig("config.yaml", dataset_path / "calibration.yaml");

    // Get the current timestamp
    auto now = std::chrono::system_clock::now();
    auto duration_since_epoch = now.time_since_epoch();
    double nsecs = static_cast<std::chrono::duration<double>>(duration_since_epoch).count();

    // Create folder for new experiment (holds the FAISS index on disk)
    std::filesystem::path folder {"examples/experiments/" + std::to_string(nsecs)};
    std::filesystem::create_directory(folder);

    rerun::RecordingStream rec("taey/clip");
    taey::connectRerun(rec, config);

    // Log a BGR cv::Mat under the given entity path (Rerun expects RGB).
    auto log_image = [&rec](const std::string &path, const cv::Mat &bgr) {
        if (bgr.empty()) return;
        cv::Mat out;
        cv::cvtColor(bgr, out,
                     bgr.channels() == 3 ? cv::COLOR_BGR2RGB
                                         : cv::COLOR_GRAY2RGB);
        rec.log(path,
                rerun::Image::from_rgb24(
                    rerun::Collection<uint8_t>::borrow(out.data, out.total() * 3),
                    {static_cast<uint32_t>(out.cols),
                     static_cast<uint32_t>(out.rows)}));
    };

    CLIP encoder(config["encoder"].as<std::string>());
    faiss::IndexFlatL2 index (512);

    // Build database
    TUM tum(dataset_path);
    std::cout << "Building FAISS Dataset" << std::endl;
    std::vector<double> timestamps = tum.timestamps();
    for (const double& timestamp : timestamps) {
        cv::Mat image = tum.getImage(timestamp);
        Eigen::VectorXf embedding = encoder(image);
        //std::vector<float> vec(embedding.data(), embedding.data() + embedding.size());
        index.add(1, embedding.data());
    }
    
    std::string index_path = std::string{folder / std::filesystem::path("dataset.index")};
    faiss::write_index(&index, index_path.data());

    // Generate timestamps
    std::size_t num_queries = config["num_queries"].as<std::size_t>();
    std::size_t step = (timestamps.size() - 1 - 0) / (num_queries - 1);
    
    std::vector<double> result(num_queries);
    std::generate(result.begin(), result.end(), 
        [current = std::size_t{0}, step, &timestamps] () mutable { 
        std::size_t value = current;
        current += step;
        return timestamps[value];
        }
    );

    int k = config["k"].as<int>();
    int query_idx = 0;
    for (const auto& timestamp : result) {
        cv::Mat image = tum.getImage(timestamp).clone();
        if (image.empty()) continue;

        // One scrubber step per query; log the query image.
        rec.set_time_sequence("query", static_cast<int64_t>(query_idx++));
        log_image("query/image", image);

        faiss::idx_t *I = new faiss::idx_t[static_cast<std::size_t>(k)];
        float *D = new float[static_cast<std::size_t>(k)];

        Eigen::VectorXf embedding = encoder(image);
        index.search(1, embedding.data(), k, D, I);

        for (faiss::idx_t i = 0; i < k; i++) {
            faiss::idx_t idx = I[i];
            if (idx < 0) continue;

            double neighbor_timestamp = timestamps[static_cast<std::size_t>(idx)];
            cv::Mat neighbor_image = tum.getImage(neighbor_timestamp);

            // Rank + distance encoded in the entity path, mirroring the old
            // on-disk filenames (e.g. neighbors/0_dist_1.234).
            log_image("query/neighbors/" + std::to_string(i) + "_dist_" +
                          std::to_string(D[i]),
                      neighbor_image);
        }

        delete[] I;
        delete[] D;
    }

    return 0;
}