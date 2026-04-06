#ifndef LOCALIZATION_COMMON__LOCALIZER_BASE_HPP_
#define LOCALIZATION_COMMON__LOCALIZER_BASE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <string>
#include <memory>

namespace autoware
{
namespace localization
{
namespace localization_common
{

/// Summary of an optimization-based registration.
struct OptimizedRegistrationSummary
{
  double score = 0.0;
  int iterations = 0;
  bool converged = false;
};

/// Base class for relative localizer ROS 2 nodes.
///
/// Template parameters:
///   ScanMsgT  — input scan message type (typically PointCloud2)
///   MapMsgT   — map message type (typically PointCloud2)
///   LocalizerT — the core localizer algorithm
///   PoseInitializerT — strategy for initial pose
template <
  typename ScanMsgT,
  typename MapMsgT,
  typename LocalizerT,
  typename PoseInitializerT>
class RelativeLocalizerNode : public rclcpp::Node
{
public:
  using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
  using Transform = geometry_msgs::msg::TransformStamped;

  RelativeLocalizerNode(
    const std::string & node_name,
    const std::string & name_space,
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : rclcpp::Node(node_name, name_space, options),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_),
    tf_broadcaster_(*this)
  {
    // Declare common parameters.
    this->declare_parameter<std::string>("map_frame", "map");
    this->declare_parameter<std::string>("base_frame", "base_link");
    this->declare_parameter<std::string>("odom_frame", "odom");

    map_frame_ = this->get_parameter("map_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    odom_frame_ = this->get_parameter("odom_frame").as_string();

    // Publishers.
    pose_pub_ = this->create_publisher<PoseWithCovarianceStamped>(
      "ndt_pose", rclcpp::QoS{10});

    // Subscribers.
    map_sub_ = this->create_subscription<MapMsgT>(
      "ndt_map", rclcpp::QoS{1}.transient_local(),
      [this](typename MapMsgT::ConstSharedPtr msg) {
        this->on_map(*msg);
      });

    scan_sub_ = this->create_subscription<ScanMsgT>(
      "points_in", rclcpp::SensorDataQoS(),
      [this](typename ScanMsgT::ConstSharedPtr msg) {
        this->on_scan(*msg);
      });

    initial_pose_sub_ = this->create_subscription<PoseWithCovarianceStamped>(
      "/localization/initialpose", rclcpp::QoS{1},
      [this](PoseWithCovarianceStamped::ConstSharedPtr msg) {
        this->on_initial_pose(*msg);
      });
  }

  virtual ~RelativeLocalizerNode() = default;

protected:
  /// Called when a new map message arrives.
  virtual void on_map(const MapMsgT & msg) = 0;

  /// Called when a new scan message arrives — run localisation here.
  virtual void on_scan(const ScanMsgT & msg) = 0;

  /// Called when an initial pose is received.
  virtual void on_initial_pose(const PoseWithCovarianceStamped & msg) = 0;

  /// Publish the localised pose and broadcast the TF.
  void publish_pose(const PoseWithCovarianceStamped & pose)
  {
    pose_pub_->publish(pose);

    // Broadcast map → base_link transform.
    Transform tf;
    tf.header = pose.header;
    tf.child_frame_id = base_frame_;
    tf.transform.translation.x = pose.pose.pose.position.x;
    tf.transform.translation.y = pose.pose.pose.position.y;
    tf.transform.translation.z = pose.pose.pose.position.z;
    tf.transform.rotation = pose.pose.pose.orientation;
    tf_broadcaster_.sendTransform(tf);
  }

  /// Validate the output of a registration. Override for custom checks.
  virtual bool validate_output(
    const OptimizedRegistrationSummary & /*summary*/,
    const PoseWithCovarianceStamped & /*pose*/)
  {
    return true;
  }

  // Accessors for subclasses.
  const std::string & map_frame() const { return map_frame_; }
  const std::string & base_frame() const { return base_frame_; }
  tf2_ros::Buffer & tf_buffer() { return tf_buffer_; }

private:
  // TF.
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  // Frame IDs.
  std::string map_frame_;
  std::string base_frame_;
  std::string odom_frame_;

  // Publishers / subscribers.
  typename rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  typename rclcpp::Subscription<MapMsgT>::SharedPtr map_sub_;
  typename rclcpp::Subscription<ScanMsgT>::SharedPtr scan_sub_;
  rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr initial_pose_sub_;
};

}  // namespace localization_common
}  // namespace localization
}  // namespace autoware

#endif  // LOCALIZATION_COMMON__LOCALIZER_BASE_HPP_
