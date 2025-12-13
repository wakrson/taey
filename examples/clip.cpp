#include <iostream>
#include <filesystem>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include "taey/TUM.h"
#include "taey/CLIP.h"
#include "taey/KeyFrame.h"

int main(int argc, char** argv) {
    // Start experiment
    auto duration_since_epoch = std::chrono::system_clock::now();.time_since_epoch();
    double timestamp = duration_since_epoch.count();

    std::filesystem::path folder {"exp_" + std::to_string(timestamp)};
    std::filesystem::create_directory(folder);

    std::filesystem::path dataset_path("datasets/rgbd_dataset_freiburg2_pioneer_slam2");
    YAML::Node config = YAML::LoadFile(dataset_path / "calibration.yaml");
    config["encoder"] = std::string{"models/encoder.clip"};

    CLIP encoder(config["encoder"]);
    faiss::IndexFlatL2 index;

    // Build database
    std::vector<double> timestamps = tum.timestamps();
    for (const double& timestamp : timestamps) {
        cv::Mat image = tum.getImage(timestamp);
        Eigen::VectorXf embedding = encoder(image);
        std::vector<float> vec(embedding.data(), embedding.data() + embedding.size());
        index.add(1, vec.data());
    }
    faiss::write_index(index, folder + "/dataset.index");

    // Generate timestamps
    int cnt = 25;
    int start = 0;
    int stop = timestamps.size()-1;
    int step = (stop - start) / (cnt - 1.0);
    
    std::vector<int> result(cnt);
    std::generate(result.begin(), result.end(), [current=start, step, timestamps] () mutable { 
        double value = current;
        current += step;
        return timestamps[value];
        }
    );

    int k = 10;
    for (const auto& timestamp : result) {
        cv::Mat image = tum.getImage(timestamp);
        Eigen::VectorXf embedding = encoder(image);
        std::vector<float> query(embedding.data(), embedding.data() + embedding.size());

        // Write query image
        std::filesystem::path subfolder {folder + "/" + std::to_string(timestamp)};
        std::filesystem::create_directories(subfolder)

        cv::imwrite(subfolder + "0.png", image);

        // Query from keyframe
        faiss::idx_t *I = new faiss::idx_t[static_cast<std::size_t>(k)];
        float *D = new float[static_cast<std::size_t>(k)];
        index.search(1, query.data(), k, D, I);

        // Initialize neighbors
        std::vector<std::shared_ptr<KeyFrame>> neighbors;
        for (faiss::idx_t i = 0; i < k; i++) {
            faiss::idx_t idx = I[i];
            float *D = new float[static_cast<std::size_t>(k)];
            double neighbor_timestamp = timestamps[static_cast<std::size_t(idx)];
            cv::Mat neighbor_image = tum.getImage(neighbor_timestamp);

            // Write neighbor image
            cv::imwrite(subfolder + "/" + std::to_string(D) + ".png", neighbor_image);
        }
    }

    return 0;
}