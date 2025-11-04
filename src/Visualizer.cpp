#include "Visualizer.h"
#include "KeyFrame.h"

#include <QHBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QVBoxLayout>
#include <opencv2/imgproc.hpp>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>

Visualizer::Visualizer(QWidget *parent) : QMainWindow(parent) {
  QWidget *central = new QWidget(this);
  QVBoxLayout *mainLayout = new QVBoxLayout(central);

  // --- Top: Point Cloud Viewer ---
  vtk_widget_ = new QVTKOpenGLNativeWidget(this);
  vtk_widget_->setMinimumHeight(300);

  // Create Qt-compatible render window and a renderer
  auto renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
  auto renderer = vtkSmartPointer<vtkRenderer>::New();

  renderWindow->AddRenderer(renderer);
  vtk_widget_->setRenderWindow(renderWindow);

  // Construct the PCL visualizer using both renderer and window
  pcl_viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>(
      renderer, renderWindow, "PointCloudViewer", false);
  pcl_viewer_->setupInteractor(vtk_widget_->interactor(),
                               pcl_viewer_->getRenderWindow());
  pcl_viewer_->setBackgroundColor(0.05, 0.05, 0.05);
  pcl_viewer_->addCoordinateSystem(0.1);

  pcl_viewer_->setCameraPosition(0.0, 0.0, -7,  // eye
                                 0.0, 0.0, 0.0, // look-at (focal point)
                                 0.0, -1.0, 0.0 // up (Y down)
  );

  cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

  // --- Bottom: RGB + Depth side by side ---
  QWidget *bottom = new QWidget(this);
  QHBoxLayout *bottomLayout = new QHBoxLayout(bottom);

  rgb_label_ = new QLabel("RGB");
  depth_label_ = new QLabel("Depth");
  rgb_label_->setAlignment(Qt::AlignCenter);
  depth_label_->setAlignment(Qt::AlignCenter);
  rgb_label_->setMinimumSize(640, 480);
  depth_label_->setMinimumSize(640, 480);

  bottomLayout->addWidget(rgb_label_);
  bottomLayout->addWidget(depth_label_);

  // --- Assemble ---
  mainLayout->addWidget(vtk_widget_);
  mainLayout->addWidget(bottom);
  setCentralWidget(central);

  resize(1400, 1000);
  setWindowTitle("TaeY Visualizer — RGB, Depth, and Point Cloud");
}

void Visualizer::showKeyFrame(std::shared_ptr<KeyFrame> kf) {
  // --- RGB image ---
  const cv::Mat &image = kf->image();
  if (!image.empty()) {
    cv::Mat rgb_img;
    if (image.channels() == 3)
      cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);
    else
      rgb_img = image.clone();

    QImage::Format format;
    if (rgb_img.channels() == 3)
      format = QImage::Format_RGB888;
    else
      format = QImage::Format_Grayscale8;

    QImage qimg(rgb_img.data, rgb_img.cols, rgb_img.rows,
                static_cast<int>(rgb_img.step), format);

    rgb_label_->setPixmap(QPixmap::fromImage(qimg).scaled(
        rgb_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
  }

  // --- Depth image ---
  const cv::Mat &depth = kf->depth();
  if (!depth.empty()) {
    cv::Mat norm;
    cv::normalize(depth, norm, 0, 255, cv::NORM_MINMAX);
    norm.convertTo(norm, CV_8U);
    QImage qimg(norm.data, norm.cols, norm.rows, static_cast<int>(norm.step),
                QImage::Format_Grayscale8);
    depth_label_->setPixmap(QPixmap::fromImage(qimg).scaled(
        depth_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  kf->objectPoints(cloud);

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);

  if (!cloud_added_) {
    pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
    pcl_viewer_->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    cloud_added_ = true;
  } else {
    pcl_viewer_->updatePointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
  }

  pcl_viewer_->spinOnce(1, true);
  vtk_widget_->renderWindow()->Render();
  vtk_widget_->update();  
}
