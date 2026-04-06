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

#include "ndt_nodes/ndt_mapping_node.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace autoware
{
namespace localization
{
namespace ndt_nodes
{

// ===========================================================================
// Constructor
// ===========================================================================

NDTMappingNode::NDTMappingNode(
  const std::string & node_name,
  const std::string & name_space,
  const rclcpp::NodeOptions & options)
: rclcpp::Node(node_name, name_space, options),
  tf_broadcaster_(*this)
{
  // -- Parameters --
  this->declare_parameter<std::string>("map_frame", "map");
  this->declare_parameter<std::string>("base_frame", "base_link");

  this->declare_parameter<double>("ndt_res", 1.0);
  this->declare_parameter<double>("step_size", 0.1);
  this->declare_parameter<double>("trans_eps", 0.01);
  this->declare_parameter<int>("max_iterations", 30);

  this->declare_parameter<double>("voxel_leaf_size", 2.0);
  this->declare_parameter<double>("min_scan_range", 5.0);
  this->declare_parameter<double>("max_scan_range", 200.0);
  this->declare_parameter<double>("min_add_scan_shift", 1.0);

  this->declare_parameter<double>("tf_x", 0.0);
  this->declare_parameter<double>("tf_y", 0.0);
  this->declare_parameter<double>("tf_z", 0.0);
  this->declare_parameter<double>("tf_roll", 0.0);
  this->declare_parameter<double>("tf_pitch", 0.0);
  this->declare_parameter<double>("tf_yaw", 0.0);

  this->declare_parameter<bool>("use_imu", false);
  this->declare_parameter<bool>("use_odom", false);
  this->declare_parameter<bool>("imu_upside_down", false);
  this->declare_parameter<bool>("enable_logging", true);

  map_frame_ = this->get_parameter("map_frame").as_string();
  base_frame_ = this->get_parameter("base_frame").as_string();

  ndt_res_ = static_cast<float>(this->get_parameter("ndt_res").as_double());
  step_size_ = this->get_parameter("step_size").as_double();
  trans_eps_ = this->get_parameter("trans_eps").as_double();
  max_iter_ = this->get_parameter("max_iterations").as_int();

  voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
  min_scan_range_ = this->get_parameter("min_scan_range").as_double();
  max_scan_range_ = this->get_parameter("max_scan_range").as_double();
  min_add_scan_shift_ = this->get_parameter("min_add_scan_shift").as_double();

  use_imu_ = this->get_parameter("use_imu").as_bool();
  use_odom_ = this->get_parameter("use_odom").as_bool();
  imu_upside_down_ = this->get_parameter("imu_upside_down").as_bool();
  enable_logging_ = this->get_parameter("enable_logging").as_bool();

  // -- base_link ↔ lidar transform --
  double tf_x = this->get_parameter("tf_x").as_double();
  double tf_y = this->get_parameter("tf_y").as_double();
  double tf_z = this->get_parameter("tf_z").as_double();
  double tf_roll = this->get_parameter("tf_roll").as_double();
  double tf_pitch = this->get_parameter("tf_pitch").as_double();
  double tf_yaw = this->get_parameter("tf_yaw").as_double();

  Eigen::Translation3f tl(
    static_cast<float>(tf_x), static_cast<float>(tf_y), static_cast<float>(tf_z));
  Eigen::AngleAxisf rot_x(static_cast<float>(tf_roll), Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf rot_y(static_cast<float>(tf_pitch), Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z(static_cast<float>(tf_yaw), Eigen::Vector3f::UnitZ());
  tf_btol_ = (tl * rot_z * rot_y * rot_x).matrix();
  tf_ltob_ = tf_btol_.inverse();

  map_cloud_.header.frame_id = map_frame_;

  // -- CSV logging --
  if (enable_logging_) {
    char buf[80];
    std::time_t now = std::time(nullptr);
    std::tm * pnow = std::localtime(&now);
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", pnow);
    std::string filename = "ndt_mapping_" + std::string(buf) + ".csv";
    log_ofs_.open(filename, std::ios::app);
    if (log_ofs_) {
      log_ofs_
        << "scan_count,stamp_sec,stamp_nanosec,frame_id,"
        << "scan_points,filtered_points,"
        << "x,y,z,roll,pitch,yaw,"
        << "iterations,fitness_score,"
        << "ndt_res,step_size,trans_eps,max_iter,"
        << "voxel_leaf_size,min_scan_range,max_scan_range,min_add_scan_shift"
        << std::endl;
      RCLCPP_INFO(this->get_logger(), "Logging to %s", filename.c_str());
    }
  }

  // -- Publishers --
  map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "ndt_map", rclcpp::QoS{1}.transient_local());
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    "current_pose", rclcpp::QoS{10});

  // -- Subscribers --
  points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "points_raw", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { on_points(msg); });

  if (use_imu_) {
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "imu_raw", rclcpp::QoS{1000},
      [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { on_imu(msg); });
  }
  if (use_odom_) {
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odom", rclcpp::QoS{1000},
      [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { on_odom(msg); });
  }

  RCLCPP_INFO(this->get_logger(), "NDTMappingNode initialized (res=%.2f, leaf=%.2f, imu=%d, odom=%d)",
    ndt_res_, voxel_leaf_size_, use_imu_, use_odom_);
}

// ===========================================================================
// Utility
// ===========================================================================

double NDTMappingNode::wrap_to_pm_pi(double angle)
{
  while (angle > M_PI) { angle -= 2.0 * M_PI; }
  while (angle < -M_PI) { angle += 2.0 * M_PI; }
  return angle;
}

double NDTMappingNode::calc_diff_for_radian(double lhs, double rhs)
{
  double diff = lhs - rhs;
  if (diff >= M_PI) { diff -= 2.0 * M_PI; }
  else if (diff < -M_PI) { diff += 2.0 * M_PI; }
  return diff;
}

NDTMappingNode::Pose6D NDTMappingNode::matrix_to_pose(const Eigen::Matrix4f & m)
{
  Pose6D p;
  p.x = static_cast<double>(m(0, 3));
  p.y = static_cast<double>(m(1, 3));
  p.z = static_cast<double>(m(2, 3));
  p.roll = std::atan2(static_cast<double>(m(2, 1)), static_cast<double>(m(2, 2)));
  p.pitch = std::asin(-static_cast<double>(m(2, 0)));
  p.yaw = std::atan2(static_cast<double>(m(1, 0)), static_cast<double>(m(0, 0)));
  return p;
}

Eigen::Matrix4f NDTMappingNode::pose_to_matrix(const Pose6D & p)
{
  float x = static_cast<float>(p.x), y = static_cast<float>(p.y), z = static_cast<float>(p.z);
  float r = static_cast<float>(p.roll), pi = static_cast<float>(p.pitch), ya = static_cast<float>(p.yaw);

  Eigen::Translation3f tl(x, y, z);
  Eigen::AngleAxisf rot_x(r, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf rot_y(pi, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z(ya, Eigen::Vector3f::UnitZ());
  return (tl * rot_z * rot_y * rot_x).matrix();
}

void NDTMappingNode::save_map(const std::string & filename, double filter_res)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(
    new pcl::PointCloud<pcl::PointXYZI>(map_cloud_));

  if (filter_res > 0.0) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> vgf;
    vgf.setLeafSize(static_cast<float>(filter_res),
                     static_cast<float>(filter_res),
                     static_cast<float>(filter_res));
    vgf.setInputCloud(map_ptr);
    vgf.filter(*filtered);
    RCLCPP_INFO(this->get_logger(), "Saving filtered map: %zu -> %zu points to %s",
      map_ptr->size(), filtered->size(), filename.c_str());
    pcl::io::savePCDFileASCII(filename, *filtered);
  } else {
    RCLCPP_INFO(this->get_logger(), "Saving map: %zu points to %s",
      map_ptr->size(), filename.c_str());
    pcl::io::savePCDFileASCII(filename, *map_ptr);
  }
}

// ===========================================================================
// IMU / Odom prediction helpers
// ===========================================================================

void NDTMappingNode::imu_calc(const rclcpp::Time & current_time)
{
  if (!imu_time_init_) {
    prev_imu_time_ = current_time;
    imu_time_init_ = true;
    return;
  }
  double dt = (current_time - prev_imu_time_).seconds();

  double diff_imu_roll = imu_msg_.angular_velocity.x * dt;
  double diff_imu_pitch = imu_msg_.angular_velocity.y * dt;
  double diff_imu_yaw = imu_msg_.angular_velocity.z * dt;

  current_pose_imu_.roll += diff_imu_roll;
  current_pose_imu_.pitch += diff_imu_pitch;
  current_pose_imu_.yaw += diff_imu_yaw;

  double accX1 = imu_msg_.linear_acceleration.x;
  double accY1 = std::cos(current_pose_imu_.roll) * imu_msg_.linear_acceleration.y -
                 std::sin(current_pose_imu_.roll) * imu_msg_.linear_acceleration.z;
  double accZ1 = std::sin(current_pose_imu_.roll) * imu_msg_.linear_acceleration.y +
                 std::cos(current_pose_imu_.roll) * imu_msg_.linear_acceleration.z;

  double accX2 = std::sin(current_pose_imu_.pitch) * accZ1 + std::cos(current_pose_imu_.pitch) * accX1;
  double accY2 = accY1;
  double accZ2 = std::cos(current_pose_imu_.pitch) * accZ1 - std::sin(current_pose_imu_.pitch) * accX1;

  double accX = std::cos(current_pose_imu_.yaw) * accX2 - std::sin(current_pose_imu_.yaw) * accY2;
  double accY = std::sin(current_pose_imu_.yaw) * accX2 + std::cos(current_pose_imu_.yaw) * accY2;
  double accZ = accZ2;

  offset_imu_x_ += current_velocity_imu_x_ * dt + accX * dt * dt / 2.0;
  offset_imu_y_ += current_velocity_imu_y_ * dt + accY * dt * dt / 2.0;
  offset_imu_z_ += current_velocity_imu_z_ * dt + accZ * dt * dt / 2.0;

  current_velocity_imu_x_ += accX * dt;
  current_velocity_imu_y_ += accY * dt;
  current_velocity_imu_z_ += accZ * dt;

  offset_imu_roll_ += diff_imu_roll;
  offset_imu_pitch_ += diff_imu_pitch;
  offset_imu_yaw_ += diff_imu_yaw;

  guess_pose_imu_.x = previous_pose_.x + offset_imu_x_;
  guess_pose_imu_.y = previous_pose_.y + offset_imu_y_;
  guess_pose_imu_.z = previous_pose_.z + offset_imu_z_;
  guess_pose_imu_.roll = previous_pose_.roll + offset_imu_roll_;
  guess_pose_imu_.pitch = previous_pose_.pitch + offset_imu_pitch_;
  guess_pose_imu_.yaw = previous_pose_.yaw + offset_imu_yaw_;

  prev_imu_time_ = current_time;
}

void NDTMappingNode::odom_calc(const rclcpp::Time & current_time)
{
  if (!odom_time_init_) {
    prev_odom_time_ = current_time;
    odom_time_init_ = true;
    return;
  }
  double dt = (current_time - prev_odom_time_).seconds();

  double diff_odom_roll = odom_msg_.twist.twist.angular.x * dt;
  double diff_odom_pitch = odom_msg_.twist.twist.angular.y * dt;
  double diff_odom_yaw = odom_msg_.twist.twist.angular.z * dt;

  current_pose_odom_.roll += diff_odom_roll;
  current_pose_odom_.pitch += diff_odom_pitch;
  current_pose_odom_.yaw += diff_odom_yaw;

  double diff_distance = odom_msg_.twist.twist.linear.x * dt;
  offset_odom_x_ += diff_distance * std::cos(-current_pose_odom_.pitch) * std::cos(current_pose_odom_.yaw);
  offset_odom_y_ += diff_distance * std::cos(-current_pose_odom_.pitch) * std::sin(current_pose_odom_.yaw);
  offset_odom_z_ += diff_distance * std::sin(-current_pose_odom_.pitch);

  offset_odom_roll_ += diff_odom_roll;
  offset_odom_pitch_ += diff_odom_pitch;
  offset_odom_yaw_ += diff_odom_yaw;

  guess_pose_odom_.x = previous_pose_.x + offset_odom_x_;
  guess_pose_odom_.y = previous_pose_.y + offset_odom_y_;
  guess_pose_odom_.z = previous_pose_.z + offset_odom_z_;
  guess_pose_odom_.roll = previous_pose_.roll + offset_odom_roll_;
  guess_pose_odom_.pitch = previous_pose_.pitch + offset_odom_pitch_;
  guess_pose_odom_.yaw = previous_pose_.yaw + offset_odom_yaw_;

  prev_odom_time_ = current_time;
}

void NDTMappingNode::imu_odom_calc(const rclcpp::Time & current_time)
{
  if (!imu_odom_time_init_) {
    prev_imu_odom_time_ = current_time;
    imu_odom_time_init_ = true;
    return;
  }
  double dt = (current_time - prev_imu_odom_time_).seconds();

  double diff_imu_roll = imu_msg_.angular_velocity.x * dt;
  double diff_imu_pitch = imu_msg_.angular_velocity.y * dt;
  double diff_imu_yaw = imu_msg_.angular_velocity.z * dt;

  current_pose_imu_odom_.roll += diff_imu_roll;
  current_pose_imu_odom_.pitch += diff_imu_pitch;
  current_pose_imu_odom_.yaw += diff_imu_yaw;

  double diff_distance = odom_msg_.twist.twist.linear.x * dt;
  offset_imu_odom_x_ += diff_distance * std::cos(-current_pose_imu_odom_.pitch) * std::cos(current_pose_imu_odom_.yaw);
  offset_imu_odom_y_ += diff_distance * std::cos(-current_pose_imu_odom_.pitch) * std::sin(current_pose_imu_odom_.yaw);
  offset_imu_odom_z_ += diff_distance * std::sin(-current_pose_imu_odom_.pitch);

  offset_imu_odom_roll_ += diff_imu_roll;
  offset_imu_odom_pitch_ += diff_imu_pitch;
  offset_imu_odom_yaw_ += diff_imu_yaw;

  guess_pose_imu_odom_.x = previous_pose_.x + offset_imu_odom_x_;
  guess_pose_imu_odom_.y = previous_pose_.y + offset_imu_odom_y_;
  guess_pose_imu_odom_.z = previous_pose_.z + offset_imu_odom_z_;
  guess_pose_imu_odom_.roll = previous_pose_.roll + offset_imu_odom_roll_;
  guess_pose_imu_odom_.pitch = previous_pose_.pitch + offset_imu_odom_pitch_;
  guess_pose_imu_odom_.yaw = previous_pose_.yaw + offset_imu_odom_yaw_;

  prev_imu_odom_time_ = current_time;
}

// ===========================================================================
// IMU callback
// ===========================================================================

void NDTMappingNode::on_imu(const sensor_msgs::msg::Imu::ConstSharedPtr & msg)
{
  sensor_msgs::msg::Imu imu = *msg;

  if (imu_upside_down_) {
    imu.angular_velocity.x *= -1.0;
    imu.angular_velocity.y *= -1.0;
    imu.angular_velocity.z *= -1.0;
    imu.linear_acceleration.x *= -1.0;
    imu.linear_acceleration.y *= -1.0;
    imu.linear_acceleration.z *= -1.0;
  }

  // Extract RPY from quaternion.
  const auto & q = imu.orientation;
  double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
  double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
  double imu_roll = std::atan2(sinr_cosp, cosr_cosp);

  double sinp = 2.0 * (q.w * q.y - q.z * q.x);
  double imu_pitch = (std::abs(sinp) >= 1.0)
    ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);

  double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  double imu_yaw = std::atan2(siny_cosp, cosy_cosp);

  imu_roll = wrap_to_pm_pi(imu_roll);
  imu_pitch = wrap_to_pm_pi(imu_pitch);
  imu_yaw = wrap_to_pm_pi(imu_yaw);

  if (!imu_orientation_init_) {
    prev_imu_roll_ = imu_roll;
    prev_imu_pitch_ = imu_pitch;
    prev_imu_yaw_ = imu_yaw;
    imu_orientation_init_ = true;
    return;
  }

  rclcpp::Time current_time = imu.header.stamp;
  double dt = imu_time_init_
    ? (current_time - prev_imu_time_).seconds()
    : 0.0;

  // Store processed IMU: derive angular velocity from orientation delta.
  imu_msg_.header = imu.header;
  imu_msg_.linear_acceleration.x = imu.linear_acceleration.x;
  imu_msg_.linear_acceleration.y = 0.0;
  imu_msg_.linear_acceleration.z = 0.0;

  if (dt > 0.0) {
    imu_msg_.angular_velocity.x = calc_diff_for_radian(imu_roll, prev_imu_roll_) / dt;
    imu_msg_.angular_velocity.y = calc_diff_for_radian(imu_pitch, prev_imu_pitch_) / dt;
    imu_msg_.angular_velocity.z = calc_diff_for_radian(imu_yaw, prev_imu_yaw_) / dt;
  } else {
    imu_msg_.angular_velocity.x = 0.0;
    imu_msg_.angular_velocity.y = 0.0;
    imu_msg_.angular_velocity.z = 0.0;
  }

  imu_calc(current_time);

  prev_imu_roll_ = imu_roll;
  prev_imu_pitch_ = imu_pitch;
  prev_imu_yaw_ = imu_yaw;
}

// ===========================================================================
// Odometry callback
// ===========================================================================

void NDTMappingNode::on_odom(const nav_msgs::msg::Odometry::ConstSharedPtr & msg)
{
  odom_msg_ = *msg;
  odom_calc(msg->header.stamp);
}

// ===========================================================================
// Points callback — main mapping loop
// ===========================================================================

void NDTMappingNode::on_points(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
{
  rclcpp::Time current_scan_time = msg->header.stamp;

  // Convert to PCL.
  pcl::PointCloud<pcl::PointXYZI> tmp;
  pcl::fromROSMsg(*msg, tmp);

  // Range filter.
  pcl::PointCloud<pcl::PointXYZI> scan;
  for (const auto & pt : tmp) {
    double r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
    if (r > min_scan_range_ && r < max_scan_range_) {
      scan.push_back(pt);
    }
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(
    new pcl::PointCloud<pcl::PointXYZI>(scan));

  // First scan: add to map directly.
  if (!initial_scan_loaded_) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed(
      new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*scan_ptr, *transformed, tf_btol_);
    map_cloud_ += *transformed;
    initial_scan_loaded_ = true;
    previous_scan_time_ = current_scan_time;
    RCLCPP_INFO(this->get_logger(), "Initial scan loaded: %zu points", map_cloud_.size());
    return;
  }

  // VoxelGrid downsample.
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan(
    new pcl::PointCloud<pcl::PointXYZI>());
  pcl::VoxelGrid<pcl::PointXYZI> vgf;
  float leaf = static_cast<float>(voxel_leaf_size_);
  vgf.setLeafSize(leaf, leaf, leaf);
  vgf.setInputCloud(scan_ptr);
  vgf.filter(*filtered_scan);

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(
    new pcl::PointCloud<pcl::PointXYZI>(map_cloud_));

  // Configure NDT.
  ndt_.setTransformationEpsilon(trans_eps_);
  ndt_.setStepSize(step_size_);
  ndt_.setResolution(ndt_res_);
  ndt_.setMaximumIterations(max_iter_);
  ndt_.setInputSource(filtered_scan);

  if (is_first_map_) {
    ndt_.setInputTarget(map_ptr);
    is_first_map_ = false;
  }

  // Compute initial guess.
  guess_pose_.x = previous_pose_.x + diff_x_;
  guess_pose_.y = previous_pose_.y + diff_y_;
  guess_pose_.z = previous_pose_.z + diff_z_;
  guess_pose_.roll = previous_pose_.roll;
  guess_pose_.pitch = previous_pose_.pitch;
  guess_pose_.yaw = previous_pose_.yaw + diff_yaw_;

  if (use_imu_ && use_odom_) { imu_odom_calc(current_scan_time); }
  if (use_imu_ && !use_odom_) { imu_calc(current_scan_time); }
  if (!use_imu_ && use_odom_) { odom_calc(current_scan_time); }

  Pose6D guess_for_ndt;
  if (use_imu_ && use_odom_) { guess_for_ndt = guess_pose_imu_odom_; }
  else if (use_imu_) { guess_for_ndt = guess_pose_imu_; }
  else if (use_odom_) { guess_for_ndt = guess_pose_odom_; }
  else { guess_for_ndt = guess_pose_; }

  Eigen::Matrix4f init_guess = pose_to_matrix(guess_for_ndt) * tf_btol_;

  // Align.
  pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(
    new pcl::PointCloud<pcl::PointXYZI>());
  ndt_.align(*output_cloud, init_guess);

  double fitness_score = ndt_.getFitnessScore();
  Eigen::Matrix4f t_localizer = ndt_.getFinalTransformation();
  bool has_converged = ndt_.hasConverged();
  int final_iterations = ndt_.getFinalNumIteration();

  Eigen::Matrix4f t_base_link = t_localizer * tf_ltob_;

  // Extract poses.
  localizer_pose_ = matrix_to_pose(t_localizer);
  ndt_pose_ = matrix_to_pose(t_base_link);
  current_pose_ = ndt_pose_;

  // Broadcast TF: map → base_link.
  geometry_msgs::msg::TransformStamped tf_msg;
  tf_msg.header.stamp = current_scan_time;
  tf_msg.header.frame_id = map_frame_;
  tf_msg.child_frame_id = base_frame_;
  tf_msg.transform.translation.x = current_pose_.x;
  tf_msg.transform.translation.y = current_pose_.y;
  tf_msg.transform.translation.z = current_pose_.z;

  Eigen::Quaternionf quat(t_base_link.block<3, 3>(0, 0));
  tf_msg.transform.rotation.x = quat.x();
  tf_msg.transform.rotation.y = quat.y();
  tf_msg.transform.rotation.z = quat.z();
  tf_msg.transform.rotation.w = quat.w();
  tf_broadcaster_.sendTransform(tf_msg);

  // Publish current_pose.
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header.stamp = current_scan_time;
  pose_msg.header.frame_id = map_frame_;
  pose_msg.pose.position.x = current_pose_.x;
  pose_msg.pose.position.y = current_pose_.y;
  pose_msg.pose.position.z = current_pose_.z;
  pose_msg.pose.orientation = tf_msg.transform.rotation;
  pose_pub_->publish(pose_msg);

  // Compute diffs.
  double secs = (current_scan_time - previous_scan_time_).seconds();
  diff_x_ = current_pose_.x - previous_pose_.x;
  diff_y_ = current_pose_.y - previous_pose_.y;
  diff_z_ = current_pose_.z - previous_pose_.z;
  diff_yaw_ = calc_diff_for_radian(current_pose_.yaw, previous_pose_.yaw);

  if (secs > 0.0) {
    current_velocity_x_ = diff_x_ / secs;
    current_velocity_y_ = diff_y_ / secs;
    current_velocity_z_ = diff_z_ / secs;
  }

  // Sync IMU/Odom pose estimates with current.
  current_pose_imu_ = current_pose_;
  current_pose_odom_ = current_pose_;
  current_pose_imu_odom_ = current_pose_;
  current_velocity_imu_x_ = current_velocity_x_;
  current_velocity_imu_y_ = current_velocity_y_;
  current_velocity_imu_z_ = current_velocity_z_;

  // Update previous pose.
  previous_pose_ = current_pose_;
  previous_scan_time_ = current_scan_time;

  // Reset offsets.
  offset_imu_x_ = offset_imu_y_ = offset_imu_z_ = 0.0;
  offset_imu_roll_ = offset_imu_pitch_ = offset_imu_yaw_ = 0.0;
  offset_odom_x_ = offset_odom_y_ = offset_odom_z_ = 0.0;
  offset_odom_roll_ = offset_odom_pitch_ = offset_odom_yaw_ = 0.0;
  offset_imu_odom_x_ = offset_imu_odom_y_ = offset_imu_odom_z_ = 0.0;
  offset_imu_odom_roll_ = offset_imu_odom_pitch_ = offset_imu_odom_yaw_ = 0.0;

  // Add scan to map if shifted enough.
  double shift = std::sqrt(
    std::pow(current_pose_.x - added_pose_.x, 2.0) +
    std::pow(current_pose_.y - added_pose_.y, 2.0));

  if (shift >= min_add_scan_shift_) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan(
      new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*scan_ptr, *transformed_scan, t_localizer);
    map_cloud_ += *transformed_scan;
    added_pose_ = current_pose_;

    // Update NDT target with new map.
    pcl::PointCloud<pcl::PointXYZI>::Ptr updated_map(
      new pcl::PointCloud<pcl::PointXYZI>(map_cloud_));
    ndt_.setInputTarget(updated_map);
  }

  // Publish accumulated map.
  sensor_msgs::msg::PointCloud2 map_msg;
  pcl::toROSMsg(map_cloud_, map_msg);
  map_msg.header.stamp = current_scan_time;
  map_msg.header.frame_id = map_frame_;
  map_pub_->publish(map_msg);

  // CSV log.
  scan_count_++;
  if (enable_logging_ && log_ofs_) {
    log_ofs_
      << scan_count_ << ","
      << msg->header.stamp.sec << "," << msg->header.stamp.nanosec << ","
      << msg->header.frame_id << ","
      << scan_ptr->size() << "," << filtered_scan->size() << ","
      << std::fixed << std::setprecision(5)
      << current_pose_.x << "," << current_pose_.y << "," << current_pose_.z << ","
      << current_pose_.roll << "," << current_pose_.pitch << "," << current_pose_.yaw << ","
      << final_iterations << "," << fitness_score << ","
      << ndt_res_ << "," << step_size_ << "," << trans_eps_ << "," << max_iter_ << ","
      << voxel_leaf_size_ << "," << min_scan_range_ << "," << max_scan_range_ << ","
      << min_add_scan_shift_ << std::endl;
  }

  RCLCPP_INFO(this->get_logger(),
    "Scan #%lu: %zu pts, %zu filtered | converged=%d iter=%d fitness=%.4f | "
    "pos=(%.2f, %.2f, %.2f) shift=%.2f map=%zu",
    scan_count_, scan_ptr->size(), filtered_scan->size(),
    has_converged, final_iterations, fitness_score,
    current_pose_.x, current_pose_.y, current_pose_.z,
    shift, map_cloud_.size());
}

}  // namespace ndt_nodes
}  // namespace localization
}  // namespace autoware
