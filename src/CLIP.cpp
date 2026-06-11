#include "CLIP.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

CLIP::CLIP(const std::string &path) { load(path); }

bool CLIP::load(const std::string &path) {
  Options options;
  options.optBatchSize = 1;
  options.maxBatchSize = 1;

  engine_ = std::make_unique<Engine<float>>(options);

  bool succ = false;
  if (std::filesystem::path(path).extension().string() == ".onnx") {
    succ = engine_->buildLoadNetwork(path, SUB_VALS, DIV_VALS, NORMALIZE);
  } else {
    succ = engine_->loadNetwork(path, SUB_VALS, DIV_VALS, NORMALIZE);
  }

  if (!succ) {
    throw std::runtime_error("Unable to load TensorRT engine: " + path);
  }
  return true;
}

Eigen::VectorXf CLIP::encode(const cv::Mat &src) {
  cv::cuda::GpuMat gpu_img;
  gpu_img.upload(src);
  return encode(gpu_img);
}

Eigen::VectorXf CLIP::encode(const cv::cuda::GpuMat &gpu_img) {
  if (engine_ == nullptr) {
    throw std::runtime_error("Image Encoding model must be loaded first!");
  }

  std::vector<std::vector<std::vector<float>>> features;
  engine_->runInference(preprocess(gpu_img), features);

  assert(features.size() == 1);
  assert(features[0].size() == 1);

  const std::vector<float> &f = features[0][0];
  Eigen::VectorXf output = Eigen::Map<const Eigen::VectorXf>(
      f.data(),
      static_cast<Eigen::Index>(f.size())
  );

  return output.normalized();
}

Eigen::VectorXf CLIP::operator()(const cv::Mat &src) {
  return encode(src);
}

std::vector<std::vector<cv::cuda::GpuMat>>
CLIP::preprocess(const cv::cuda::GpuMat &gpu_img) {
  const auto &inputDims = engine_->getInputDims();

  cv::cuda::cvtColor(gpu_img, rgb_mat_, cv::COLOR_BGR2RGB);
  cv::cuda::resize(rgb_mat_, resized_,
                   cv::Size(static_cast<int>(inputDims[0].d[1]),
                            static_cast<int>(inputDims[0].d[2])));

  std::vector<cv::cuda::GpuMat> input{resized_};
  std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

  return inputs;
}

double CLIP::cosineDistance(const Eigen::VectorXf &x1,
                            const Eigen::VectorXf &x2) {
  double dot = static_cast<double>(x1.dot(x2));
  dot = std::clamp(dot, -1.0, 1.0);
  return 1.0 - dot;
}
