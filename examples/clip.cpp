#include <iostream>
#include <filesystem>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include "taey/TUM.h"
#include "taey/CLIP.h"
#include "taey/KeyFrame.h"

int main() {
    // Get the curre timestamp
    auto now = std::chrono::system_clock::now();
    auto duration_since_epoch = now.time_since_epoch();
    double nsecs = static_cast<std::chrono::duration<double>>(duration_since_epoch).count();

    // Create folder for new experiment
    std::filesystem::path folder {"exp_" + std::to_string(nsecs)};
    std::filesystem::create_directory(folder);

    std::filesystem::path dataset_path("datasets/rgbd_dataset_freiburg2_pioneer_slam2");
    YAML::Node config = YAML::LoadFile(dataset_path / "calibration.yaml");
    config["encoder"] = std::string{"models/encoder.clip"};

    CLIP encoder(config["encoder"].as<std::string>());
    faiss::IndexFlatL2 index;

    // Build database
    TUM tum(dataset_path);
    std::vector<double> timestamps = tum.timestamps();
    for (const double& timestamp : timestamps) {
        cv::Mat image = tum.getImage(timestamp);
        Eigen::VectorXf embedding = encoder(image);
        std::vector<float> vec(embedding.data(), embedding.data() + embedding.size());
        index.add(1, vec.data());
    }
    std::string index_path = std::string{folder / std::filesystem::path("dataset.index")};
    faiss::write_index(&index, index_path.data());

    // Generate timestamps
    std::size_t cnt = 25;
    std::size_t step = (timestamps.size() - 1 - 0) / (cnt - 1);
    
    std::vector<int> result(cnt);
    std::generate(result.begin(), result.end(), [current=0, step, timestamps] () mutable { 
        std::size_t value = current;
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
        std::filesystem::path subfolder {folder.string() + "/" + std::to_string(timestamp)};
        std::filesystem::create_directories(subfolder);

        cv::imwrite(subfolder.string() + "0.png", image);

        // Query from keyframe
        faiss::idx_t *I = new faiss::idx_t[static_cast<std::size_t>(k)];
        float *D = new float[static_cast<std::size_t>(k)];
        index.search(1, query.data(), k, D, I);

        // Initialize neighbors
        std::vector<std::shared_ptr<KeyFrame>> neighbors;
        for (faiss::idx_t i = 0; i < k; i++) {
            faiss::idx_t idx = I[i];
            float distance = D[i];
            double neighbor_timestamp = timestamps[static_cast<std::size_t>(idx)];
            cv::Mat neighbor_image = tum.getImage(neighbor_timestamp);

            // Write neighbor image
            cv::imwrite(subfolder.string() + "/" + std::to_string(distance) + ".png", neighbor_image);
        }
        delete[] I;
        delete[] D;
    }

    return 0;
}