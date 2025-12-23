#include "Visualizer.h"
#include "KeyFrame.h" // Ensure this header has the correct methods

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QMetaType> // Crucial for passing custom types across threads

#include <opencv2/imgproc.hpp>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>
#include <pcl/filters/voxel_grid.h> // For optimizing map size

Visualizer::Visualizer(QWidget *parent) : QMainWindow(parent) {
    // 1. Register types so they can be passed via Signals
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>>("std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>");

    // 2. Connect the internal signal to the GUI update slot
    // This ensures that even if addKeyFrame is called from a background thread, 
    // slotRenderData runs safely on the main thread.
    connect(this, &Visualizer::signalUpdateGUI, this, &Visualizer::slotRenderData);

    // --- UI Setup ---
    QWidget *central = new QWidget(this);
    QVBoxLayout *mainLayout = new QVBoxLayout(central);

    // Setup VTK Widget
    vtk_widget_ = new QVTKOpenGLNativeWidget(this);
    vtk_widget_->setMinimumHeight(300);
    
    auto renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    renderWindow->AddRenderer(renderer);
    vtk_widget_->setRenderWindow(renderWindow);

    pcl_viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>(
        renderer, renderWindow, "Viewer", false);
    
    pcl_viewer_->setBackgroundColor(0.1, 0.1, 0.1); // Dark grey looks better than pitch black
    pcl_viewer_->addCoordinateSystem(0.5);
    pcl_viewer_->setCameraPosition(0, -5, -5, 0, 0, 0, 0, -1, 0);

    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Setup Labels
    QWidget *bottom = new QWidget(this);
    QHBoxLayout *bottomLayout = new QHBoxLayout(bottom);
    
    rgb_label_ = new QLabel("RGB");
    depth_label_ = new QLabel("Depth");
    rgb_label_->setScaledContents(true); // Allow automatic scaling
    depth_label_->setScaledContents(true);
    
    // Set fixed aspect ratio policies if desired, or min size
    rgb_label_->setMinimumSize(320, 240);
    depth_label_->setMinimumSize(320, 240);

    bottomLayout->addWidget(rgb_label_);
    bottomLayout->addWidget(depth_label_);

    mainLayout->addWidget(vtk_widget_, 1); // 1 = give viewer more stretch priority
    mainLayout->addWidget(bottom, 0);
    
    setCentralWidget(central);
    resize(1200, 900);
    setWindowTitle("Realtime SLAM Visualizer");
}

// --- THREAD SAFE ENTRY POINT ---
// This function runs on the SLAM thread
void Visualizer::addKeyFrame(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return;

cv::Mat rgb = kf->image().clone();
    cv::Mat depth = kf->depth().clone();

    // 2. Compute Raw Point Cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_points(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // Assuming objectPoints fills the cloud (likely in World Coordinates)
    kf->objectPoints(raw_points); 

    // 3. FILTER THE POINTS
    // Create a new container for the clean points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clean_points(new pcl::PointCloud<pcl::PointXYZRGB>);
    clean_points->points.reserve(raw_points->points.size()); // Pre-allocate for speed

    for (const auto& pt : raw_points->points) {
        // Filter 1: Remove Invalid Values (NaN/Inf)
        // These crash the visualizer or cause "exploding" rendering artifacts
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            continue;
        }

        // Filter 2: Remove Origin Artifacts (0,0,0)
        // Invalid depth pixels often project exactly to (0,0,0). 
        // In a map, this looks like a dense cluster or "triangle tip" at the origin.
        if (pt.x == 0.0f && pt.y == 0.0f && pt.z == 0.0f) {
            continue;
        }

        // Filter 3: (Optional) Y-Height or Range Ceiling
        // If your drone/robot shouldn't see points 20m high, filter them here.
        // if (pt.y > 5.0f || pt.y < -5.0f) continue;

        clean_points->points.push_back(pt);
    }

    // Update metadata for the new cloud
    clean_points->width = clean_points->points.size();
    clean_points->height = 1;
    clean_points->is_dense = true;

    // 3. Emit Signal to Main Thread
    emit signalUpdateGUI(rgb, depth, clean_points);
}

// --- GUI UPDATE SLOT ---
// This function runs on the Main Qt Thread
void Visualizer::slotRenderData(cv::Mat rgb, cv::Mat depth, 
                                std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> new_points) {
    
    // A. Update RGB Label
    if (!rgb.empty()) {
        if (rgb.channels() == 3) cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
        QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        rgb_label_->setPixmap(QPixmap::fromImage(qimg));
    }

    // B. Update Depth Label
    if (!depth.empty()) {
        cv::Mat norm_depth;
        cv::normalize(depth, norm_depth, 0, 255, cv::NORM_MINMAX);
        norm_depth.convertTo(norm_depth, CV_8U);
        QImage qdepth(norm_depth.data, norm_depth.cols, norm_depth.rows, norm_depth.step, QImage::Format_Grayscale8);
        depth_label_->setPixmap(QPixmap::fromImage(qdepth));
    }

    // C. Update Point Cloud (With Downsampling for Realtime Performance)
    if (new_points && !new_points->empty()) {
        
        // Accumulate points
        *map_cloud_ += *new_points;

        // OPTIMIZATION: Voxel Grid Filter
        // If map gets too large (> 50k points), filter it to maintain FPS
        if (map_cloud_->size() > 50000) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::VoxelGrid<pcl::PointXYZRGB> sor;
            sor.setInputCloud(map_cloud_);
            sor.setLeafSize(0.05f, 0.05f, 0.05f); // 5cm voxel size
            sor.filter(*filtered_cloud);
            map_cloud_ = filtered_cloud;
        }

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler(map_cloud_);
        
        if (is_first_frame_) {
            pcl_viewer_->addPointCloud<pcl::PointXYZRGB>(map_cloud_, rgb_handler, "map_cloud");
            pcl_viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "map_cloud");
            is_first_frame_ = false;
        } else {
            pcl_viewer_->updatePointCloud<pcl::PointXYZRGB>(map_cloud_, rgb_handler, "map_cloud");
        }

        // TRIGGER RENDER
        // Do NOT use spinOnce in Qt. Use the widget's update.
        vtk_widget_->renderWindow()->Render();
        vtk_widget_->update(); 
    }
}