#include <chrono>

#include "FramePoint.h"
#include "Camera.h"
#include "CLIP.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "Optimizer.h"
#include "TAEY.h"
#include "Visualizer.h"

#include <QMetaType>

Q_DECLARE_METATYPE(std::shared_ptr<KeyFrame>)

TAEY::TAEY(int &argc, char **argv, const YAML::Node &config)
    : QApplication(argc, argv) {
  qRegisterMetaType<std::shared_ptr<KeyFrame>>("std::shared_ptr<KeyFrame>");

  emit_pending_ = false;
  running_ = true;
  config_ = config;
  map_ = std::make_shared<Map>(config);
  optimizer_ = std::make_shared<Optimizer>(map_, config_);

  vit_ = std::make_unique<CLIP>(config_["encoder"].as<std::string>());
  vis_ = std::make_unique<Visualizer>();
  vis_->moveToThread(thread());
  vis_->show();

  //QObject::connect(this, &TAEY::keyFrameReady, vis_.get(),
  //                 &Visualizer::showKeyFrame, Qt::QueuedConnection);
}

void TAEY::reset() {
  emit_pending_ = false;
  running_ = true;
  map_ = std::make_shared<Map>(config_);
  optimizer_ = std::make_shared<Optimizer>(map_, config_);
}

TAEY::~TAEY() {
  running_ = false;
  cv_.notify_all();
  if (vis_thread_.joinable()) {
    vis_thread_.join();
  }
}

std::shared_ptr<Map> TAEY::map() const { return map_; }

std::shared_ptr<KeyFrame> TAEY::operator()(const cv::Mat &image,
                                           const cv::Mat &depth) {
  auto now = std::chrono::system_clock::now();
  // Calculate the duration since the epoch (usually Unix epoch: January 1,
  // 1970)
  auto duration_since_epoch = now.time_since_epoch();
  // Convert the duration to a duration type with a double representation and
  // seconds as the unit std::chrono::duration<double> represents a duration in
  // seconds with double precision
  std::chrono::duration<double> seconds_since_epoch = duration_since_epoch;
  double timestamp = seconds_since_epoch.count();

  // Get keyframe id
  std::size_t key_frame_id = map_->numKeyFrames();
  // Create key frame from new RGBD data
  std::shared_ptr<KeyFrame> key_frame = std::make_shared<KeyFrame>(
      key_frame_id, timestamp, image, depth, config_);

  // Set frame points
  for (auto &fp : key_frame->framePoints()) {
    fp->setKeyFrame(key_frame);
  }

  // Extract features
  Eigen::VectorXf features = vit_->encode(key_frame->image());
  key_frame->imageEmbedding(features);

  // Check if the new key frame is valid (aka has frame points)
  bool status = false;
  if (key_frame->numFramePoints() > 0) {
    status = track(key_frame);
    // Update visualizer
    if (status == true && key_frame) {
      // Update visualizer
      if (!emit_pending_) {
        emit_pending_ = true;
        QMetaObject::invokeMethod(
            vis_.get(),
            [this, key_frame]() {
              vis_->addKeyFrame(key_frame);
              emit_pending_ = false;
            },
            Qt::QueuedConnection);
      }
    } else {
      key_frame.reset();
    }
  }
  return key_frame;
}

bool TAEY::track(std::shared_ptr<KeyFrame> &key_frame) {
  // Track the frame points to existing map points or create new map points
  // Get matches between the current frame's frame points and the other
  // keyframe's frame points

  // First insert
  std::vector<std::shared_ptr<MapPoint>> mps(key_frame->framePoints().size(),
                                             nullptr);
  if (map_->numKeyFrames() == 0) {
    for (std::size_t i = 0; i < key_frame->framePoints().size(); i++) {
      // Create new map points with new pose
      // Backproject to get image point in world frame
      Eigen::MatrixXd pW = key_frame->camera()->backProject(
          key_frame->framePoints()[i], Camera::Frame::WORLD);
      // Create new map point
      mps[i] = std::make_shared<MapPoint>(i, key_frame->framePoints()[i],
                                          pW.transpose());
      // Set frame points map point (weak link)
      key_frame->framePoints()[i]->setMapPoint(mps[i]);
    }
    map_->insert(key_frame);
    map_->insert(mps);
    optimizer_->update(key_frame->id());
    return true;
  }

  std::vector<std::shared_ptr<MapPoint>> matches = map_->track(key_frame);

  int track_cnt = static_cast<int>(
      std::count_if(matches.begin(), matches.end(),
                    [](const auto &m) { return m != nullptr; }));

  if (track_cnt < 20) {
    return false;
  }

  // Hold all map points
  std::vector<std::shared_ptr<MapPoint>> map_points(matches.size(), nullptr);
  // Loop through each match and insert new track (FramePoint)
  std::size_t num_matches = 0;
  for (std::size_t i = 0; i < matches.size(); i++) {
    // match found
    if (matches[i] != nullptr) {
      // Set MapPoint in FramePoint
      key_frame->framePoints()[i]->setMapPoint(matches[i]);
      map_points[i] = matches[i];
      num_matches++;
    }
  }

  // Estimate pose with matches
  bool status = true;
  if (map_->numKeyFrames() > 0) {
    Eigen::Transform<double, 3, Eigen::Isometry> pose;
    status = key_frame->estimatePose(pose);
    key_frame->setPose(pose);
  }

  // Estimated valid pose
  if (status == true) {
    // Get the number of map points
    std::size_t num_map_points = map_->numMapPoints();
    // Create new map points with new pose
    for (std::size_t i = 0; i < matches.size(); i++) {
      // No matching map point (create a new one)
      if (matches[i] == nullptr) {
        // Backproject to get image point in world frame
        Eigen::MatrixXd pW = key_frame->camera()->backProject(
            key_frame->framePoints()[i], Camera::Frame::WORLD);
        // Create new map point
        map_points[i] = std::make_shared<MapPoint>(
            num_map_points, key_frame->framePoints()[i], pW.transpose());
        // Set frame points map point (weak link)
        key_frame->framePoints()[i]->setMapPoint(map_points[i]);
        num_map_points++;
      }
      // Insert FramePoint into MapPoint
      else {
        matches[i]->insert(key_frame->framePoints()[i]);
      }
    }
    // Add to map
    // std::cout << "inserting : " << key_frame->id() << std::endl;
    map_->insert(key_frame);
    map_->insert(map_points);
    optimizer_->update(key_frame->id());
  }
  // delete
  else {
    key_frame.reset();
  }
  return status;
}