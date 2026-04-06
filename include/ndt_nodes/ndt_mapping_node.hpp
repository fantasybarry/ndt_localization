// Copyright 2015-2019 Autoware Foundation. All rights reserved.
// Ported to ROS 2 from ndt_mapping.cpp by Yuki Kitsukawa.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NDT_NODES__NDT_MAPPING_NODE_HPP_
#define NDT_NODES__NDT_MAPPING_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>

#include <string>
#include <fstream>

namespace autoware
{
namespace localization
{
namespace ndt_nodes
{

class NDTMappingNode : public rclcpp::Node
{
public:
  explicit NDTMappingNode(
    const std::string & node_name = "ndt_mapping",
    const std::string & name_space = "localization",
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  struct Pose6D
  {
    double x = 0.0, y = 0.0, z = 0.0;
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
  };

  // ---- Callbacks ----
  void on_points(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg);
  void on_imu(const sensor_msgs::msg::Imu::ConstSharedPtr & msg);
  void on_odom(const nav_msgs::msg::Odometry::ConstSharedPtr & msg);

  // ---- Prediction helpers ----
  void imu_calc(const rclcpp::Time & current_time);
  void odom_calc(const rclcpp::Time & current_time);
  void imu_odom_calc(const rclcpp::Time & current_time);

  // ---- Utility ----
  static double wrap_to_pm_pi(double angle);
  static double calc_diff_for_radian(double lhs, double rhs);
  static Pose6D matrix_to_pose(const Eigen::Matrix4f & m);
  static Eigen::Matrix4f pose_to_matrix(const Pose6D & p);
  void save_map(const std::string & filename, double filter_res);

  // ---- NDT ----
  pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt_;
  pcl::PointCloud<pcl::PointXYZI> map_cloud_;

  // ---- NDT parameters ----
  int max_iter_;
  float ndt_res_;
  double step_size_;
  double trans_eps_;
  double voxel_leaf_size_;
  double min_scan_range_;
  double max_scan_range_;
  double min_add_scan_shift_;

  // ---- Pose state ----
  Pose6D previous_pose_{};
  Pose6D current_pose_{};
  Pose6D added_pose_{};
  Pose6D guess_pose_{};
  Pose6D guess_pose_imu_{};
  Pose6D guess_pose_odom_{};
  Pose6D guess_pose_imu_odom_{};
  Pose6D current_pose_imu_{};
  Pose6D current_pose_odom_{};
  Pose6D current_pose_imu_odom_{};
  Pose6D ndt_pose_{};
  Pose6D localizer_pose_{};

  // ---- Velocity state ----
  double current_velocity_x_{}, current_velocity_y_{}, current_velocity_z_{};
  double current_velocity_imu_x_{}, current_velocity_imu_y_{}, current_velocity_imu_z_{};

  // ---- Diff state ----
  double diff_x_{}, diff_y_{}, diff_z_{}, diff_yaw_{};

  // ---- IMU/Odom offset accumulators ----
  double offset_imu_x_{}, offset_imu_y_{}, offset_imu_z_{};
  double offset_imu_roll_{}, offset_imu_pitch_{}, offset_imu_yaw_{};
  double offset_odom_x_{}, offset_odom_y_{}, offset_odom_z_{};
  double offset_odom_roll_{}, offset_odom_pitch_{}, offset_odom_yaw_{};
  double offset_imu_odom_x_{}, offset_imu_odom_y_{}, offset_imu_odom_z_{};
  double offset_imu_odom_roll_{}, offset_imu_odom_pitch_{}, offset_imu_odom_yaw_{};

  // ---- Cached sensor messages ----
  sensor_msgs::msg::Imu imu_msg_;
  nav_msgs::msg::Odometry odom_msg_;

  // ---- Timing for prediction ----
  rclcpp::Time prev_imu_time_;
  rclcpp::Time prev_odom_time_;
  rclcpp::Time prev_imu_odom_time_;
  rclcpp::Time previous_scan_time_;
  bool imu_time_init_{false};
  bool odom_time_init_{false};
  bool imu_odom_time_init_{false};

  // ---- Previous IMU orientation ----
  double prev_imu_roll_{}, prev_imu_pitch_{}, prev_imu_yaw_{};
  bool imu_orientation_init_{false};

  // ---- base_link ↔ lidar transform ----
  Eigen::Matrix4f tf_btol_;
  Eigen::Matrix4f tf_ltob_;

  // ---- Feature flags ----
  bool use_imu_;
  bool use_odom_;
  bool imu_upside_down_;
  bool enable_logging_;

  // ---- Scan tracking ----
  bool initial_scan_loaded_{false};
  bool is_first_map_{true};
  uint64_t scan_count_{0};

  // ---- Frame IDs ----
  std::string map_frame_;
  std::string base_frame_;

  // ---- ROS 2 plumbing ----
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  // ---- CSV logging ----
  std::ofstream log_ofs_;
};

}  // namespace ndt_nodes
}  // namespace localization
}  // namespace autoware

#endif  // NDT_NODES__NDT_MAPPING_NODE_HPP_
