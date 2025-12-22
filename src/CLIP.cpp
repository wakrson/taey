#include "CLIP.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "engine.h"

CLIP::CLIP(const std::string &path) { load(path); }

bool CLIP::load(const std::string &path) {
  Options options;
  options.optBatchSize = 1;
  options.maxBatchSize = 1;
  if (options.precision == Precision::INT8) {
    if (options.calibrationDataDirectoryPath.empty()) {
      throw std::runtime_error(
          "Error: Must supply calibration data path for INT8 calibration");
    }
  }

  // Create our TensorRT inference engine
  engine_ = std::make_unique<Engine<float>>(options);
  // Build onnx model
  if (std::filesystem::path(path).extension().string() == "onnx") {
    bool succ = engine_->buildLoadNetwork(path, SUB_VALS, DIV_VALS,
                                                    NORMALIZE);
    if (!succ) {
      throw std::runtime_error("Unable to build TensorRT engine.");
    }
  }
  // Load tensorrt model
  else {
    bool succ =
        engine_->loadNetwork(path, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
      throw std::runtime_error("Unable to load TensorRT engine.");
    }
  }
  return true;
}

Eigen::VectorXf CLIP::encode(const cv::Mat &src) {
  if (engine_ == nullptr) {
    throw std::runtime_error("Image Encoding model must be loaded first!");
  }
  cv::cuda::GpuMat gpu_img;
  gpu_img.upload(src);

  // Extract features
  std::vector<std::vector<std::vector<float>>> features;
  engine_->runInference(preprocess(gpu_img), features);

  // Ensure dhape
  assert(features.size() == 1);
  assert(features[0].size() == 1 || features[0].size() == 512);

  // Convert to eigen
  const std::vector<float> &f = features[0][0];
  Eigen::VectorXf output = Eigen::Map<const Eigen::VectorXf>(
      f.data(),
      static_cast<Eigen::Index>(f.size())
  );

  // Normalize vector
  //float norm = output.norm();
  //if (norm > 1e-12f) {
  //  output /= norm;
  //}
  return output.normalized();
}

Eigen::VectorXf CLIP::operator()(const cv::Mat &src) {
  return encode(src);
}

std::vector<std::vector<cv::cuda::GpuMat>>
CLIP::preprocess(const cv::cuda::GpuMat &gpu_img) {
  // Populate the input vectors
  const auto &inputDims = engine_->getInputDims();

  // Convert the image from BGR to RGB
  cv::cuda::GpuMat rgb_mat;
  cv::cuda::cvtColor(gpu_img, rgb_mat, cv::COLOR_BGR2RGB);

  cv::cuda::GpuMat resized;
  cv::cuda::resize(rgb_mat, resized,
                   cv::Size(static_cast<int>(inputDims[0].d[1]),
                            static_cast<int>(inputDims[0].d[2])));

  // Convert to format expected by our inference engine
  // The reason for the strange format is because it supports models with
  // multiple inputs as well as batching In our case though, the model only has
  // a single input and we are using a batch size of 1.
  std::vector<cv::cuda::GpuMat> input{std::move(resized)};
  std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

  // These params will be used in the post-processing stage

  return inputs;
}

double CLIP::cosineDistance(const Eigen::VectorXf &x1,
                            const Eigen::VectorXf &x2) {
  /*
  // Compute dot product
  double dot = x1.dot(x2);

  // Compute norms
  double norm1 = x1.norm();
  double norm2 = x2.norm();

  if (norm1 == 0.0 || norm2 == 0.0) {
    throw std::invalid_argument("Vectors must be non-zero");
  }

  // Cosine similarity
  double cos_sim = dot / (norm1 * norm2);

  // Cosine distance = 1 - similarity
  return 1.0 - cos_sim;
  */
  double dot = static_cast<double>(x1.dot(x2));

  // Floating point noise can result in 1.0000001, which can break 
  // logic elsewhere (like acos). Clamp to [-1, 1].
  if (dot > 1.0) dot = 1.0;
  if (dot < -1.0) dot = -1.0;

  // Distance = 1 - Similarity
  return 1.0 - dot;
}