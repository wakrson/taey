#include "rclcpp/rclcpp.hpp"

#include <chrono>
#include <functional>
#include <memory>

#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"

#include <sensor_msgs/msg/Image.hpp>

using namespace std::chrono_literals;

using std::placeholders::_1;
using std::placeholders::_2;

class TAEYNode : public rclcpp::Node {
public:
    TAEYNode() : Node("taey") {
        rclcpp::QoS qos = rclcpp::QoS(10);

        rgb_sub_.subscribe(this, "temp", qos);
        depth_sub_.subscribe(this, "fluid", qos);

        timer_ = this->create_wall_timer(500ms, std::bind(&TAEYNode::TimerCallback, this));
        second_timer = this->create_wall_timer(550ms, std::bind(&TAEYNode::SecondTimerCallback, this));

        uint32_t queue_size = 10;
        sync_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::
            ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>>(
            message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
            sensor_msgs::msg::Image>(queue_size), rgb_sub_, depth_sub_);

        sync_->setAgePenalty(0.50);
        sync_->registerCallback(std::bind(&TAEYNode::SyncCallback, this, _1, _2));

        std::unique_ptr<TAEY> taey_ = std::make_unique<TAEY>();
    }
  private:
    void SyncCallback(const sensor_msgs::msg::Image::ConstSharedPtr & rgb_img,
        const sensor_msgs::msg::Image::ConstSharedPtr & depth_img) {
        RCLCPP_INFO(this->get_logger(), "Sync callback with %u and %u as times",
            rgb_img->header.stamp.sec, depth_img->header.stamp.sec);
        
    }

    void TimerCallback() {
        std::cout << "TIMER CALLBACK" << std::endl;
    }

    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TAEYNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}