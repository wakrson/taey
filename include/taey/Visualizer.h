#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QVBoxLayout>
#include <memory>
#include <opencv2/opencv.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <string>

#include <QVTKOpenGLNativeWidget.h> // <-- use VTK + Qt widget

class KeyFrame;

class Visualizer : public QMainWindow {
  Q_OBJECT
public:
  explicit Visualizer(QWidget *parent = nullptr);

public slots:
  void showKeyFrame(std::shared_ptr<KeyFrame> kf);

private:
  QLabel *rgb_label_;
  QLabel *depth_label_;
  QVTKOpenGLNativeWidget *vtk_widget_;
  std::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_, map_cloud_;
  bool cloud_added_ = false;
};
