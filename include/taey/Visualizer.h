#pragma once
#include <QMainWindow>
#include <QLabel>
#include <memory>
#include <opencv2/core.hpp>

// PCL & VTK
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <QVTKOpenGLNativeWidget.h>

// Forward declaration
class KeyFrame;

class Visualizer : public QMainWindow {
    Q_OBJECT

public:
    explicit Visualizer(QWidget *parent = nullptr);
    ~Visualizer() = default;

    // This is the function your SLAM thread calls
    void addKeyFrame(std::shared_ptr<KeyFrame> kf);

signals:
    // Internal signal to pass data to the main GUI thread
    void signalUpdateGUI(cv::Mat rgb, cv::Mat depth, 
                         std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> new_points);

private slots:
    // The slot that actually touches the widgets (runs on Main Thread)
    void slotRenderData(cv::Mat rgb, cv::Mat depth, 
                        std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> new_points);

private:
    // UI Elements
    QVTKOpenGLNativeWidget* vtk_widget_;
    std::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer_;
    QLabel* rgb_label_;
    QLabel* depth_label_;

    // Data
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr map_cloud_;
    bool is_first_frame_ = true;
};