// Copyright 2019 the Autoware Foundation
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
//
// Co-developed by Tier IV, Inc. and Apex.AI, Inc.

#ifndef NDT_NODES__NDT_LOCALIZER_NODES_HPP_
#define NDT_NODES__NDT_LOCALIZER_NODES_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int32.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <string>
#include <memory>
#include <limits>
#include <cmath>
#include <chrono>

namespace autoware
{
namespace localization
{
namespace ndt_nodes
{

/// P2D NDT Localizer ROS 2 Node (Autoware-compatible interface).
///
/// Incorporates features from both Autoware.Auto and Autoware.AI ndt_matching:
///   - Custom P2D NDT with Hessian-based covariance estimation
///   - IMU + Odometry fusion for initial guess prediction
///   - GNSS-based initialization and re-initialization
///   - base_link ↔ lidar static transform
///   - Velocity/twist estimation and publishing
///   - NDT reliability metric
///
/// Input:
///   - ekf_pose_with_covariance           (PoseWithCovarianceStamped)  — initial pose
///   - pointcloud_map                     (PointCloud2, transient local) — map pointcloud
///   - points_raw                         (PointCloud2, sensor QoS)     — sensor pointcloud
///   - sensing/gnss/pose_with_covariance  (PoseWithCovarianceStamped)  — GNSS regularization
///   - imu_raw                            (Imu)                        — IMU for prediction
///   - odom                               (Odometry)                   — odometry for prediction
///
/// Output:
///   - ndt_pose                           (PoseStamped)                — estimated pose
///   - ndt_pose_with_covariance           (PoseWithCovarianceStamped)  — with covariance
///   - diagnostics                        (DiagnosticArray)            — diagnostics
///   - estimate_twist                     (TwistStamped)               — estimated velocity
///   - exe_time_ms                        (Float32)                    — [debug] execution time
///   - transform_probability              (Float64)                    — [debug] score
///   - iteration_num                      (Int32)                      — [debug] iterations
///   - ndt_reliability                    (Float32)                    — [debug] reliability metric
///   - initial_to_result_distance         (Float32)                    — [debug] init→result dist
///   - predict_pose                       (PoseStamped)                — [debug] predicted pose
///   - initial_pose_with_covariance       (PoseWithCovarianceStamped)  — [debug] initial pose used
///   - estimated_vel_mps                  (Float32)                    — [debug] velocity m/s
///   - points_aligned                     (PointCloud2)                — [debug] aligned scan
///   - marker                             (MarkerArray)                — visualization
///   - monte_carlo_initial_pose           (PoseArray)                  — [debug] MC particles
///   - TF: map → base_link
class P2DNDTLocalizerNode : public rclcpp::Node
{
public:
  using CloudT = sensor_msgs::msg::PointCloud2;
  using PoseStamped = geometry_msgs::msg::PoseStamped;
  using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
  using PoseArray = geometry_msgs::msg::PoseArray;
  using TwistStamped = geometry_msgs::msg::TwistStamped;
  using Transform = geometry_msgs::msg::TransformStamped;

  P2DNDTLocalizerNode(
    const std::string & node_name,
    const std::string & name_space)
  : rclcpp::Node(node_name, name_space),
    tf_broadcaster_(*this)
  {
    // -- Parameters --
    this->declare_parameter<std::string>("map_frame", "map");
    this->declare_parameter<std::string>("base_frame", "base_link");
    this->declare_parameter<int>("max_iterations", 30);
    this->declare_parameter<double>("step_size", 1.0);
    this->declare_parameter<double>("epsilon", 1e-4);
    this->declare_parameter<double>("score_threshold", 2.0);
    this->declare_parameter<double>("ndt_resolution", 1.0);
    this->declare_parameter<double>("scan_voxel_leaf_size", 2.0);
    this->declare_parameter<double>("predict_pose_threshold", 15.0);

    // IMU / Odom / GNSS feature flags.
    this->declare_parameter<bool>("use_imu", false);
    this->declare_parameter<bool>("use_odom", false);
    this->declare_parameter<bool>("imu_upside_down", false);
    this->declare_parameter<double>("gnss_reinit_fitness", 500.0);

    // base_link → lidar static transform.
    this->declare_parameter<double>("tf_x", 0.0);
    this->declare_parameter<double>("tf_y", 0.0);
    this->declare_parameter<double>("tf_z", 0.0);
    this->declare_parameter<double>("tf_roll", 0.0);
    this->declare_parameter<double>("tf_pitch", 0.0);
    this->declare_parameter<double>("tf_yaw", 0.0);

    // NDT reliability weights.
    this->declare_parameter<double>("reliability_wa", 0.4);
    this->declare_parameter<double>("reliability_wb", 0.3);
    this->declare_parameter<double>("reliability_wc", 0.3);

    map_frame_ = this->get_parameter("map_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    score_threshold_ = this->get_parameter("score_threshold").as_double();
    scan_voxel_leaf_size_ = this->get_parameter("scan_voxel_leaf_size").as_double();
    predict_pose_threshold_ = this->get_parameter("predict_pose_threshold").as_double();

    use_imu_ = this->get_parameter("use_imu").as_bool();
    use_odom_ = this->get_parameter("use_odom").as_bool();
    imu_upside_down_ = this->get_parameter("imu_upside_down").as_bool();
    gnss_reinit_fitness_ = this->get_parameter("gnss_reinit_fitness").as_double();

    Wa_ = this->get_parameter("reliability_wa").as_double();
    Wb_ = this->get_parameter("reliability_wb").as_double();
    Wc_ = this->get_parameter("reliability_wc").as_double();

    // Build base_link ↔ lidar transform.
    {
      float tx = static_cast<float>(this->get_parameter("tf_x").as_double());
      float ty = static_cast<float>(this->get_parameter("tf_y").as_double());
      float tz = static_cast<float>(this->get_parameter("tf_z").as_double());
      float r  = static_cast<float>(this->get_parameter("tf_roll").as_double());
      float p  = static_cast<float>(this->get_parameter("tf_pitch").as_double());
      float y  = static_cast<float>(this->get_parameter("tf_yaw").as_double());
      Eigen::Translation3f tl(tx, ty, tz);
      Eigen::AngleAxisf rx(r, Eigen::Vector3f::UnitX());
      Eigen::AngleAxisf ry(p, Eigen::Vector3f::UnitY());
      Eigen::AngleAxisf rz(y, Eigen::Vector3f::UnitZ());
      tf_btol_ = (tl * rz * ry * rx).matrix();
      tf_ltob_ = tf_btol_.inverse();
      has_lidar_tf_ = (tx != 0.0f || ty != 0.0f || tz != 0.0f ||
                        r != 0.0f || p != 0.0f || y != 0.0f);
    }

    // Build PCL NDT.
    ndt_.setMaximumIterations(this->get_parameter("max_iterations").as_int());
    ndt_.setStepSize(this->get_parameter("step_size").as_double());
    ndt_.setTransformationEpsilon(this->get_parameter("epsilon").as_double());
    ndt_.setResolution(static_cast<float>(this->get_parameter("ndt_resolution").as_double()));

    // -- Publishers (output) --
    ndt_pose_pub_ = this->create_publisher<PoseStamped>(
      "ndt_pose", rclcpp::QoS{10});
    ndt_pose_with_cov_pub_ = this->create_publisher<PoseWithCovarianceStamped>(
      "ndt_pose_with_covariance", rclcpp::QoS{10});
    diagnostics_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "diagnostics", rclcpp::QoS{10});
    estimate_twist_pub_ = this->create_publisher<TwistStamped>(
      "estimate_twist", rclcpp::QoS{10});
    predict_pose_pub_ = this->create_publisher<PoseStamped>(
      "predict_pose", rclcpp::QoS{10});
    points_aligned_pub_ = this->create_publisher<CloudT>(
      "points_aligned", rclcpp::QoS{10});
    initial_pose_with_cov_pub_ = this->create_publisher<PoseWithCovarianceStamped>(
      "initial_pose_with_covariance", rclcpp::QoS{10});
    exe_time_ms_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "exe_time_ms", rclcpp::QoS{10});
    transform_probability_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "transform_probability", rclcpp::QoS{10});
    no_ground_transform_probability_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "no_ground_transform_probability", rclcpp::QoS{10});
    iteration_num_pub_ = this->create_publisher<std_msgs::msg::Int32>(
      "iteration_num", rclcpp::QoS{10});
    ndt_reliability_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "ndt_reliability", rclcpp::QoS{10});
    initial_to_result_distance_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "initial_to_result_distance", rclcpp::QoS{10});
    initial_to_result_distance_old_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "initial_to_result_distance_old", rclcpp::QoS{10});
    initial_to_result_distance_old_2d_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "initial_to_result_distance_old_2d", rclcpp::QoS{10});
    initial_to_result_relative_pose_pub_ = this->create_publisher<PoseStamped>(
      "initial_to_result_relative_pose", rclcpp::QoS{10});
    nvtl_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "nearest_voxel_transformation_likelihood", rclcpp::QoS{10});
    no_ground_nvtl_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "no_ground_nearest_voxel_transformation_likelihood", rclcpp::QoS{10});
    estimated_vel_mps_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "estimated_vel_mps", rclcpp::QoS{10});
    estimated_vel_kmph_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "estimated_vel_kmph", rclcpp::QoS{10});
    multi_ndt_pose_pub_ = this->create_publisher<PoseArray>(
      "multi_ndt_pose", rclcpp::QoS{10});
    multi_initial_pose_pub_ = this->create_publisher<PoseArray>(
      "multi_initial_pose", rclcpp::QoS{10});
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "marker", rclcpp::QoS{10});
    monte_carlo_initial_pose_pub_ = this->create_publisher<PoseArray>(
      "monte_carlo_initial_pose", rclcpp::QoS{10});

    // -- Subscribers (input) --
    ekf_pose_sub_ = this->create_subscription<PoseWithCovarianceStamped>(
      "ekf_pose_with_covariance", rclcpp::QoS{1},
      [this](PoseWithCovarianceStamped::ConstSharedPtr msg) { on_initial_pose(*msg); });

    map_sub_ = this->create_subscription<CloudT>(
      "pointcloud_map", rclcpp::QoS{1}.transient_local(),
      [this](CloudT::ConstSharedPtr msg) { on_map(*msg); });

    scan_sub_ = this->create_subscription<CloudT>(
      "points_raw", rclcpp::SensorDataQoS(),
      [this](CloudT::ConstSharedPtr msg) { on_scan(*msg); });

    gnss_pose_sub_ = this->create_subscription<PoseWithCovarianceStamped>(
      "sensing/gnss/pose_with_covariance", rclcpp::QoS{1},
      [this](PoseWithCovarianceStamped::ConstSharedPtr msg) { on_gnss_pose(*msg); });

    if (use_imu_) {
      imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "imu_raw", rclcpp::QoS{1000},
        [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { on_imu(*msg); });
    }
    if (use_odom_) {
      odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odom", rclcpp::QoS{1000},
        [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { on_odom(*msg); });
    }

    RCLCPP_INFO(this->get_logger(),
      "P2DNDTLocalizerNode initialized (imu=%d, odom=%d, lidar_tf=%d)",
      use_imu_, use_odom_, has_lidar_tf_);
  }

private:
  // ---- Pose struct for prediction ----
  struct Pose6D
  {
    double x = 0.0, y = 0.0, z = 0.0;
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
  };

  // ---- Callbacks ----

  void on_map(const CloudT & msg)
  {
    if (map_received_) { return; }
    RCLCPP_INFO(this->get_logger(), "Received map with %u points", msg.width);
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(msg, *map_cloud);

    // Remove NaN points.
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*map_cloud, *map_cloud, indices);

    RCLCPP_INFO(this->get_logger(), "Setting NDT target with %zu points", map_cloud->size());
    ndt_.setInputTarget(map_cloud);
    map_received_ = true;
  }

  void on_initial_pose(const PoseWithCovarianceStamped & msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received initial pose (EKF)");
    latest_pose_msg_ = msg;
    init_pose_received_ = true;
    first_ndt_accepted_ = false;

    // Reset pose state from the initial pose.
    previous_pose_ = pose_msg_to_6d(msg);
    current_pose_ = previous_pose_;
    current_pose_imu_ = previous_pose_;
    current_pose_odom_ = previous_pose_;
    current_pose_imu_odom_ = previous_pose_;

    // Reset velocities and offsets.
    current_velocity_x_ = current_velocity_y_ = current_velocity_z_ = 0.0;
    angular_velocity_ = 0.0;
    current_velocity_imu_x_ = current_velocity_imu_y_ = current_velocity_imu_z_ = 0.0;
    reset_offsets();
  }

  void on_gnss_pose(const PoseWithCovarianceStamped & msg)
  {
    gnss_pose_ = msg;
    gnss_received_ = true;

    // GNSS re-initialization when fitness degrades.
    if (!init_pose_received_ || last_fitness_score_ >= gnss_reinit_fitness_) {
      RCLCPP_WARN(this->get_logger(),
        "GNSS re-init (fitness=%.2f, threshold=%.2f)", last_fitness_score_, gnss_reinit_fitness_);
      on_initial_pose(msg);
    }
  }

  void on_imu(const sensor_msgs::msg::Imu & msg)
  {
    // Don't accumulate IMU offsets until we have an initial pose and first scan time.
    if (!init_pose_received_) { return; }

    sensor_msgs::msg::Imu imu = msg;

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
    double imu_roll = std::atan2(
      2.0 * (q.w * q.x + q.y * q.z),
      1.0 - 2.0 * (q.x * q.x + q.y * q.y));
    double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    double imu_pitch = (std::abs(sinp) >= 1.0)
      ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);
    double imu_yaw = std::atan2(
      2.0 * (q.w * q.z + q.x * q.y),
      1.0 - 2.0 * (q.y * q.y + q.z * q.z));

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

    rclcpp::Time t = imu.header.stamp;
    double dt = imu_time_init_ ? (t - prev_imu_time_).seconds() : 0.0;

    // Derive angular velocity from orientation delta (as in ndt_matching).
    imu_msg_.header = imu.header;
    imu_msg_.linear_acceleration.x = imu.linear_acceleration.x;
    imu_msg_.linear_acceleration.y = 0.0;
    imu_msg_.linear_acceleration.z = 0.0;

    if (dt > 0.0) {
      imu_msg_.angular_velocity.x = calc_diff_for_radian(imu_roll, prev_imu_roll_) / dt;
      imu_msg_.angular_velocity.y = calc_diff_for_radian(imu_pitch, prev_imu_pitch_) / dt;
      imu_msg_.angular_velocity.z = calc_diff_for_radian(imu_yaw, prev_imu_yaw_) / dt;
    } else {
      imu_msg_.angular_velocity.x = imu_msg_.angular_velocity.y = imu_msg_.angular_velocity.z = 0.0;
    }

    imu_calc(t);

    prev_imu_time_ = t;
    imu_time_init_ = true;
    prev_imu_roll_ = imu_roll;
    prev_imu_pitch_ = imu_pitch;
    prev_imu_yaw_ = imu_yaw;
  }

  void on_odom(const nav_msgs::msg::Odometry & msg)
  {
    odom_msg_ = msg;
    odom_calc(msg.header.stamp);
  }

  void on_scan(const CloudT & msg)
  {
    auto t_start = std::chrono::steady_clock::now();
    rclcpp::Time current_scan_time = msg.header.stamp;

    if (!map_received_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "No map received yet, skipping scan");
      return;
    }
    if (!init_pose_received_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "No initial pose received yet, skipping scan");
      return;
    }

    // Downsample scan with VoxelGrid filter.
    pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(msg, *scan_cloud);

    // Remove NaN points.
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*scan_cloud, *scan_cloud, indices);

    if (scan_voxel_leaf_size_ > 0.0) {
      pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
      voxel_filter.setInputCloud(scan_cloud);
      float leaf = static_cast<float>(scan_voxel_leaf_size_);
      voxel_filter.setLeafSize(leaf, leaf, leaf);
      pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
      voxel_filter.filter(*filtered);
      scan_cloud = filtered;
    }

    // -- Compute initial guess with IMU/Odom/linear prediction --
    double diff_time = scan_time_init_
      ? (current_scan_time - previous_scan_time_).seconds() : 0.0;

    // Linear extrapolation from velocity.
    Pose6D predict_pose;
    predict_pose.x = previous_pose_.x + current_velocity_x_ * diff_time;
    predict_pose.y = previous_pose_.y + current_velocity_y_ * diff_time;
    predict_pose.z = previous_pose_.z + current_velocity_z_ * diff_time;
    predict_pose.roll = previous_pose_.roll;
    predict_pose.pitch = previous_pose_.pitch;
    predict_pose.yaw = previous_pose_.yaw + angular_velocity_ * diff_time;

    // Run IMU/Odom prediction if enabled (only after first scan processed).
    Pose6D predict_for_ndt;
    if (!scan_time_init_) {
      // First scan: use previous pose directly (no prediction).
      predict_for_ndt = previous_pose_;
      // Reset IMU/Odom accumulators so they start fresh from this scan.
      reset_offsets();
      imu_time_init_ = false;
      odom_time_init_ = false;
    } else {
      if (use_imu_ && use_odom_) { imu_odom_calc(current_scan_time); }
      else if (use_imu_) { imu_calc(current_scan_time); }
      else if (use_odom_) { odom_calc(current_scan_time); }

      if (use_imu_ && use_odom_) { predict_for_ndt = predict_pose_imu_odom_; }
      else if (use_imu_) { predict_for_ndt = predict_pose_imu_; }
      else if (use_odom_) { predict_for_ndt = predict_pose_odom_; }
      else { predict_for_ndt = predict_pose; }
    }

    // Publish predict_pose (debug).
    {
      PoseStamped pm;
      pm.header.stamp = current_scan_time;
      pm.header.frame_id = map_frame_;
      pose6d_to_msg(predict_for_ndt, pm.pose);
      predict_pose_pub_->publish(pm);
    }

    // Build initial guess as 4x4 matrix.
    Eigen::Matrix4f initial_guess = pose6d_to_matrix4f(predict_for_ndt);

    // Publish initial pose used (debug).
    {
      PoseWithCovarianceStamped ip;
      ip.header.stamp = current_scan_time;
      ip.header.frame_id = map_frame_;
      pose6d_to_msg(predict_for_ndt, ip.pose.pose);
      initial_pose_with_cov_pub_->publish(ip);
    }

    // -- Run PCL NDT alignment --
    ndt_.setInputSource(scan_cloud);
    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    ndt_.align(aligned_cloud, initial_guess);

    auto t_end = std::chrono::steady_clock::now();
    double exe_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    bool converged = ndt_.hasConverged();
    int iterations = ndt_.getFinalNumIteration();
    double fitness_score = ndt_.getFitnessScore();
    double transform_probability = ndt_.getTransformationProbability();

    last_fitness_score_ = fitness_score;

    RCLCPP_DEBUG(this->get_logger(),
      "NDT iter=%d fitness=%.4f prob=%.4f converged=%d exe=%.1fms pts=%zu",
      iterations, fitness_score, transform_probability, converged, exe_time,
      scan_cloud->size());

    if (!converged) {
      RCLCPP_WARN(this->get_logger(), "NDT did not converge (iter=%d)", iterations);
      publish_diagnostics("NDT did not converge", false);
      return;
    }

    // Reject poor matches based on fitness score (mean squared nearest-neighbor distance).
    if (fitness_score > score_threshold_) {
      RCLCPP_WARN(this->get_logger(),
        "NDT fitness too high (%.2f > %.2f), rejecting", fitness_score, score_threshold_);
      publish_diagnostics("NDT fitness too high", false);
      return;
    }

    // -- Extract result pose --
    Eigen::Matrix4f result_matrix = ndt_.getFinalTransformation();
    Pose6D ndt_result = matrix4f_to_pose6d(result_matrix);

    // Log distance from prediction for diagnostics.
    double dist_predict = std::sqrt(
      std::pow(ndt_result.x - predict_for_ndt.x, 2) +
      std::pow(ndt_result.y - predict_for_ndt.y, 2) +
      std::pow(ndt_result.z - predict_for_ndt.z, 2));

    // Predict-pose threshold: reject if NDT result deviates too far from prediction.
    // Skip on first scan — initial pose may be approximate.
    if (first_ndt_accepted_ && dist_predict > predict_pose_threshold_) {
      RCLCPP_WARN(this->get_logger(),
        "NDT result too far from prediction (%.2f > %.2f), rejecting",
        dist_predict, predict_pose_threshold_);
      publish_diagnostics("NDT result rejected (too far from prediction)", false);
      return;
    }

    current_pose_ = ndt_result;
    if (!first_ndt_accepted_) {
      RCLCPP_INFO(this->get_logger(), "First NDT lock-on at (%.2f, %.2f, %.2f)",
        ndt_result.x, ndt_result.y, ndt_result.z);
      first_ndt_accepted_ = true;
    }

    RCLCPP_DEBUG(this->get_logger(),
      "NDT pose: (%.3f, %.3f, %.3f) rpy=(%.3f, %.3f, %.3f) dist_predict=%.2f",
      ndt_result.x, ndt_result.y, ndt_result.z,
      ndt_result.roll, ndt_result.pitch, ndt_result.yaw, dist_predict);

    // Covariance estimate: scale diagonal by fitness score.
    // Lower fitness = better match = smaller covariance.
    double cov_scale = std::max(fitness_score, 0.01);
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity() * cov_scale;

    // -- Publish ndt_pose (PoseStamped) --
    PoseStamped pose_stamped;
    pose_stamped.header.stamp = current_scan_time;
    pose_stamped.header.frame_id = map_frame_;
    pose6d_to_msg(current_pose_, pose_stamped.pose);
    ndt_pose_pub_->publish(pose_stamped);

    // -- Publish ndt_pose_with_covariance --
    PoseWithCovarianceStamped pose_cov;
    pose_cov.header = pose_stamped.header;
    pose_cov.pose.pose = pose_stamped.pose;
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        pose_cov.pose.covariance[static_cast<size_t>(r * 6 + c)] = cov(r, c);
      }
    }
    ndt_pose_with_cov_pub_->publish(pose_cov);

    // -- Publish TF: map → base_link --
    Transform tf_msg;
    tf_msg.header = pose_stamped.header;
    tf_msg.child_frame_id = base_frame_;
    tf_msg.transform.translation.x = current_pose_.x;
    tf_msg.transform.translation.y = current_pose_.y;
    tf_msg.transform.translation.z = current_pose_.z;
    tf_msg.transform.rotation = pose_stamped.pose.orientation;
    tf_broadcaster_.sendTransform(tf_msg);

    // -- Compute velocity and twist --
    double diff_x = current_pose_.x - previous_pose_.x;
    double diff_y = current_pose_.y - previous_pose_.y;
    double diff_z = current_pose_.z - previous_pose_.z;
    double diff_yaw = calc_diff_for_radian(current_pose_.yaw, previous_pose_.yaw);
    double diff_3d = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

    if (diff_time > 0.0) {
      current_velocity_x_ = diff_x / diff_time;
      current_velocity_y_ = diff_y / diff_time;
      current_velocity_z_ = diff_z / diff_time;
      angular_velocity_ = diff_yaw / diff_time;
      current_velocity_ = diff_3d / diff_time;
    }

    // Publish estimate_twist.
    TwistStamped twist;
    twist.header.stamp = current_scan_time;
    twist.header.frame_id = base_frame_;
    twist.twist.linear.x = current_velocity_;
    twist.twist.linear.y = 0.0;
    twist.twist.linear.z = 0.0;
    twist.twist.angular.x = 0.0;
    twist.twist.angular.y = 0.0;
    twist.twist.angular.z = angular_velocity_;
    estimate_twist_pub_->publish(twist);

    // Publish estimated velocities.
    std_msgs::msg::Float32 vel_mps, vel_kmph;
    vel_mps.data = static_cast<float>(current_velocity_);
    vel_kmph.data = static_cast<float>(current_velocity_ * 3.6);
    estimated_vel_mps_pub_->publish(vel_mps);
    estimated_vel_kmph_pub_->publish(vel_kmph);

    // -- Debug topics --

    // exe_time_ms
    std_msgs::msg::Float32 exe_msg;
    exe_msg.data = static_cast<float>(exe_time);
    exe_time_ms_pub_->publish(exe_msg);

    // transform_probability (NDT score)
    std_msgs::msg::Float64 score_msg;
    score_msg.data = transform_probability;
    transform_probability_pub_->publish(score_msg);
    no_ground_transform_probability_pub_->publish(score_msg);
    nvtl_pub_->publish(score_msg);
    no_ground_nvtl_pub_->publish(score_msg);

    // iteration_num
    std_msgs::msg::Int32 iter_msg;
    iter_msg.data = iterations;
    iteration_num_pub_->publish(iter_msg);

    // NDT reliability: Wa*(exe/100)*100 + Wb*(iter/10)*100 + Wc*((2-prob)/2)*100
    double trans_prob = std::min(fitness_score, 2.0);
    std_msgs::msg::Float32 rel_msg;
    rel_msg.data = static_cast<float>(
      Wa_ * (exe_time / 100.0) * 100.0 +
      Wb_ * (static_cast<double>(iterations) / 10.0) * 100.0 +
      Wc_ * ((2.0 - trans_prob) / 2.0) * 100.0);
    ndt_reliability_pub_->publish(rel_msg);

    // initial_to_result_distance (using Pose6D values)
    double dist_3d = std::sqrt(
      std::pow(ndt_result.x - predict_for_ndt.x, 2) +
      std::pow(ndt_result.y - predict_for_ndt.y, 2) +
      std::pow(ndt_result.z - predict_for_ndt.z, 2));
    double dist_2d = std::sqrt(
      std::pow(ndt_result.x - predict_for_ndt.x, 2) +
      std::pow(ndt_result.y - predict_for_ndt.y, 2));

    std_msgs::msg::Float32 dist_msg;
    dist_msg.data = static_cast<float>(dist_3d);
    initial_to_result_distance_pub_->publish(dist_msg);
    initial_to_result_distance_old_pub_->publish(dist_msg);

    std_msgs::msg::Float32 dist_2d_msg;
    dist_2d_msg.data = static_cast<float>(dist_2d);
    initial_to_result_distance_old_2d_pub_->publish(dist_2d_msg);

    // initial_to_result_relative_pose
    {
      PoseStamped rp;
      rp.header = pose_stamped.header;
      Pose6D rel;
      rel.x = ndt_result.x - predict_for_ndt.x;
      rel.y = ndt_result.y - predict_for_ndt.y;
      rel.z = ndt_result.z - predict_for_ndt.z;
      rel.roll = ndt_result.roll - predict_for_ndt.roll;
      rel.pitch = ndt_result.pitch - predict_for_ndt.pitch;
      rel.yaw = ndt_result.yaw - predict_for_ndt.yaw;
      pose6d_to_msg(rel, rp.pose);
      initial_to_result_relative_pose_pub_->publish(rp);
    }

    // diagnostics
    publish_diagnostics("OK", true);

    // -- Update state for next iteration --
    current_pose_imu_ = current_pose_;
    current_pose_odom_ = current_pose_;
    current_pose_imu_odom_ = current_pose_;
    current_velocity_imu_x_ = current_velocity_x_;
    current_velocity_imu_y_ = current_velocity_y_;
    current_velocity_imu_z_ = current_velocity_z_;

    previous_pose_ = current_pose_;
    previous_scan_time_ = current_scan_time;
    scan_time_init_ = true;

    reset_offsets();
  }

  // ---- IMU/Odom prediction helpers (ported from ndt_matching.cpp) ----

  void imu_calc(const rclcpp::Time & current_time)
  {
    if (!imu_time_init_) {
      prev_imu_time_ = current_time;
      imu_time_init_ = true;
      return;
    }
    double dt = (current_time - prev_imu_time_).seconds();
    if (dt <= 0.0 || dt > 1.0) {
      prev_imu_time_ = current_time;
      return;
    }

    // Only use angular velocity for orientation prediction.
    // Skip acceleration double-integration to avoid gravity-induced drift.
    double dy = imu_msg_.angular_velocity.z * dt;
    offset_imu_yaw_ += dy;

    // Use velocity from previous NDT match for position extrapolation.
    offset_imu_x_ += current_velocity_x_ * dt;
    offset_imu_y_ += current_velocity_y_ * dt;
    offset_imu_z_ += current_velocity_z_ * dt;

    predict_pose_imu_.x = previous_pose_.x + offset_imu_x_;
    predict_pose_imu_.y = previous_pose_.y + offset_imu_y_;
    predict_pose_imu_.z = previous_pose_.z + offset_imu_z_;
    predict_pose_imu_.roll = previous_pose_.roll;
    predict_pose_imu_.pitch = previous_pose_.pitch;
    predict_pose_imu_.yaw = previous_pose_.yaw + offset_imu_yaw_;

    prev_imu_time_ = current_time;
  }

  void odom_calc(const rclcpp::Time & current_time)
  {
    if (!odom_time_init_) {
      prev_odom_time_ = current_time;
      odom_time_init_ = true;
      return;
    }
    double dt = (current_time - prev_odom_time_).seconds();
    if (dt <= 0.0) { return; }

    double dr = odom_msg_.twist.twist.angular.x * dt;
    double dp = odom_msg_.twist.twist.angular.y * dt;
    double dy = odom_msg_.twist.twist.angular.z * dt;

    current_pose_odom_.roll += dr;
    current_pose_odom_.pitch += dp;
    current_pose_odom_.yaw += dy;

    double dist = odom_msg_.twist.twist.linear.x * dt;
    offset_odom_x_ += dist * std::cos(-current_pose_odom_.pitch) * std::cos(current_pose_odom_.yaw);
    offset_odom_y_ += dist * std::cos(-current_pose_odom_.pitch) * std::sin(current_pose_odom_.yaw);
    offset_odom_z_ += dist * std::sin(-current_pose_odom_.pitch);

    offset_odom_roll_ += dr;
    offset_odom_pitch_ += dp;
    offset_odom_yaw_ += dy;

    predict_pose_odom_.x = previous_pose_.x + offset_odom_x_;
    predict_pose_odom_.y = previous_pose_.y + offset_odom_y_;
    predict_pose_odom_.z = previous_pose_.z + offset_odom_z_;
    predict_pose_odom_.roll = previous_pose_.roll + offset_odom_roll_;
    predict_pose_odom_.pitch = previous_pose_.pitch + offset_odom_pitch_;
    predict_pose_odom_.yaw = previous_pose_.yaw + offset_odom_yaw_;

    prev_odom_time_ = current_time;
  }

  void imu_odom_calc(const rclcpp::Time & current_time)
  {
    if (!imu_odom_time_init_) {
      prev_imu_odom_time_ = current_time;
      imu_odom_time_init_ = true;
      return;
    }
    double dt = (current_time - prev_imu_odom_time_).seconds();
    if (dt <= 0.0) { return; }

    double dr = imu_msg_.angular_velocity.x * dt;
    double dp = imu_msg_.angular_velocity.y * dt;
    double dy = imu_msg_.angular_velocity.z * dt;

    current_pose_imu_odom_.roll += dr;
    current_pose_imu_odom_.pitch += dp;
    current_pose_imu_odom_.yaw += dy;

    double dist = odom_msg_.twist.twist.linear.x * dt;
    offset_imu_odom_x_ += dist * std::cos(-current_pose_imu_odom_.pitch) * std::cos(current_pose_imu_odom_.yaw);
    offset_imu_odom_y_ += dist * std::cos(-current_pose_imu_odom_.pitch) * std::sin(current_pose_imu_odom_.yaw);
    offset_imu_odom_z_ += dist * std::sin(-current_pose_imu_odom_.pitch);

    offset_imu_odom_roll_ += dr;
    offset_imu_odom_pitch_ += dp;
    offset_imu_odom_yaw_ += dy;

    predict_pose_imu_odom_.x = previous_pose_.x + offset_imu_odom_x_;
    predict_pose_imu_odom_.y = previous_pose_.y + offset_imu_odom_y_;
    predict_pose_imu_odom_.z = previous_pose_.z + offset_imu_odom_z_;
    predict_pose_imu_odom_.roll = previous_pose_.roll + offset_imu_odom_roll_;
    predict_pose_imu_odom_.pitch = previous_pose_.pitch + offset_imu_odom_pitch_;
    predict_pose_imu_odom_.yaw = previous_pose_.yaw + offset_imu_odom_yaw_;

    prev_imu_odom_time_ = current_time;
  }

  void reset_offsets()
  {
    offset_imu_x_ = offset_imu_y_ = offset_imu_z_ = 0.0;
    offset_imu_roll_ = offset_imu_pitch_ = offset_imu_yaw_ = 0.0;
    offset_odom_x_ = offset_odom_y_ = offset_odom_z_ = 0.0;
    offset_odom_roll_ = offset_odom_pitch_ = offset_odom_yaw_ = 0.0;
    offset_imu_odom_x_ = offset_imu_odom_y_ = offset_imu_odom_z_ = 0.0;
    offset_imu_odom_roll_ = offset_imu_odom_pitch_ = offset_imu_odom_yaw_ = 0.0;
  }

  // ---- Helpers ----

  void publish_diagnostics(const std::string & message, bool ok)
  {
    diagnostic_msgs::msg::DiagnosticArray diag_array;
    diag_array.header.stamp = this->now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "ndt_scan_matcher";
    status.level = ok
      ? diagnostic_msgs::msg::DiagnosticStatus::OK
      : diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message = message;
    diag_array.status.push_back(status);
    diagnostics_pub_->publish(diag_array);
  }

  static double wrap_to_pm_pi(double angle)
  {
    while (angle > M_PI) { angle -= 2.0 * M_PI; }
    while (angle < -M_PI) { angle += 2.0 * M_PI; }
    return angle;
  }

  static double calc_diff_for_radian(double lhs, double rhs)
  {
    double d = lhs - rhs;
    if (d >= M_PI) { d -= 2.0 * M_PI; }
    else if (d < -M_PI) { d += 2.0 * M_PI; }
    return d;
  }

  static Pose6D pose_msg_to_6d(const PoseWithCovarianceStamped & msg)
  {
    Pose6D p;
    p.x = msg.pose.pose.position.x;
    p.y = msg.pose.pose.position.y;
    p.z = msg.pose.pose.position.z;
    const auto & q = msg.pose.pose.orientation;
    p.roll = std::atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y));
    double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    p.pitch = (std::abs(sinp) >= 1.0) ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);
    p.yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    return p;
  }

  static Eigen::Matrix4f pose6d_to_matrix4f(const Pose6D & p)
  {
    Eigen::AngleAxisf rx(static_cast<float>(p.roll), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf ry(static_cast<float>(p.pitch), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rz(static_cast<float>(p.yaw), Eigen::Vector3f::UnitZ());
    Eigen::Translation3f tl(
      static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
    return (tl * rz * ry * rx).matrix();
  }

  static Pose6D matrix4f_to_pose6d(const Eigen::Matrix4f & m)
  {
    Pose6D p;
    p.x = static_cast<double>(m(0, 3));
    p.y = static_cast<double>(m(1, 3));
    p.z = static_cast<double>(m(2, 3));
    // Extract Euler angles (ZYX convention).
    p.roll = static_cast<double>(std::atan2(m(2, 1), m(2, 2)));
    p.pitch = static_cast<double>(std::asin(-m(2, 0)));
    p.yaw = static_cast<double>(std::atan2(m(1, 0), m(0, 0)));
    return p;
  }

  static void pose6d_to_msg(const Pose6D & p, geometry_msgs::msg::Pose & msg)
  {
    msg.position.x = p.x;
    msg.position.y = p.y;
    msg.position.z = p.z;
    const double cr = std::cos(p.roll * 0.5), sr = std::sin(p.roll * 0.5);
    const double cp = std::cos(p.pitch * 0.5), sp = std::sin(p.pitch * 0.5);
    const double cy = std::cos(p.yaw * 0.5), sy = std::sin(p.yaw * 0.5);
    msg.orientation.w = cr * cp * cy + sr * sp * sy;
    msg.orientation.x = sr * cp * cy - cr * sp * sy;
    msg.orientation.y = cr * sp * cy + sr * cp * sy;
    msg.orientation.z = cr * cp * sy - sr * sp * cy;
  }

  // ---- Members ----

  // Core NDT (PCL).
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  // base_link ↔ lidar transform.
  Eigen::Matrix4f tf_btol_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f tf_ltob_ = Eigen::Matrix4f::Identity();
  bool has_lidar_tf_ = false;

  // Pose prediction state.
  Pose6D previous_pose_{};
  Pose6D current_pose_{};
  Pose6D predict_pose_imu_{};
  Pose6D predict_pose_odom_{};
  Pose6D predict_pose_imu_odom_{};
  Pose6D current_pose_imu_{};
  Pose6D current_pose_odom_{};
  Pose6D current_pose_imu_odom_{};

  // Velocity state.
  double current_velocity_ = 0.0;
  double current_velocity_x_ = 0.0, current_velocity_y_ = 0.0, current_velocity_z_ = 0.0;
  double angular_velocity_ = 0.0;
  double current_velocity_imu_x_ = 0.0, current_velocity_imu_y_ = 0.0, current_velocity_imu_z_ = 0.0;

  // IMU/Odom offset accumulators.
  double offset_imu_x_ = 0, offset_imu_y_ = 0, offset_imu_z_ = 0;
  double offset_imu_roll_ = 0, offset_imu_pitch_ = 0, offset_imu_yaw_ = 0;
  double offset_odom_x_ = 0, offset_odom_y_ = 0, offset_odom_z_ = 0;
  double offset_odom_roll_ = 0, offset_odom_pitch_ = 0, offset_odom_yaw_ = 0;
  double offset_imu_odom_x_ = 0, offset_imu_odom_y_ = 0, offset_imu_odom_z_ = 0;
  double offset_imu_odom_roll_ = 0, offset_imu_odom_pitch_ = 0, offset_imu_odom_yaw_ = 0;

  // Cached sensor messages.
  sensor_msgs::msg::Imu imu_msg_;
  nav_msgs::msg::Odometry odom_msg_;
  PoseWithCovarianceStamped latest_pose_msg_;
  PoseWithCovarianceStamped gnss_pose_;

  // Timing for prediction.
  rclcpp::Time prev_imu_time_;
  rclcpp::Time prev_odom_time_;
  rclcpp::Time prev_imu_odom_time_;
  rclcpp::Time previous_scan_time_;
  bool imu_time_init_ = false;
  bool odom_time_init_ = false;
  bool imu_odom_time_init_ = false;
  bool scan_time_init_ = false;

  // IMU orientation state.
  double prev_imu_roll_ = 0, prev_imu_pitch_ = 0, prev_imu_yaw_ = 0;
  bool imu_orientation_init_ = false;

  // Feature flags.
  bool use_imu_ = false;
  bool use_odom_ = false;
  bool imu_upside_down_ = false;
  double gnss_reinit_fitness_ = 500.0;
  double last_fitness_score_ = 0.0;

  // Reliability weights.
  double Wa_ = 0.4, Wb_ = 0.3, Wc_ = 0.3;

  // Subscribers.
  rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr ekf_pose_sub_;
  rclcpp::Subscription<CloudT>::SharedPtr map_sub_;
  rclcpp::Subscription<CloudT>::SharedPtr scan_sub_;
  rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr gnss_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  // Publishers.
  rclcpp::Publisher<PoseStamped>::SharedPtr ndt_pose_pub_;
  rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr ndt_pose_with_cov_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::Publisher<TwistStamped>::SharedPtr estimate_twist_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr predict_pose_pub_;
  rclcpp::Publisher<CloudT>::SharedPtr points_aligned_pub_;
  rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr initial_pose_with_cov_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr exe_time_ms_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr transform_probability_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr no_ground_transform_probability_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr iteration_num_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr ndt_reliability_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr initial_to_result_distance_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr initial_to_result_distance_old_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr initial_to_result_distance_old_2d_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr initial_to_result_relative_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr nvtl_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr no_ground_nvtl_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr estimated_vel_mps_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr estimated_vel_kmph_pub_;
  rclcpp::Publisher<PoseArray>::SharedPtr multi_ndt_pose_pub_;
  rclcpp::Publisher<PoseArray>::SharedPtr multi_initial_pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<PoseArray>::SharedPtr monte_carlo_initial_pose_pub_;

  // Config.
  std::string map_frame_;
  std::string base_frame_;
  bool map_received_ = false;
  bool init_pose_received_ = false;
  bool first_ndt_accepted_ = false;
  bool gnss_received_ = false;
  double score_threshold_ = 2.0;
  double scan_voxel_leaf_size_ = 2.0;
  double predict_pose_threshold_ = 15.0;
};

}  // namespace ndt_nodes
}  // namespace localization
}  // namespace autoware

#endif  // NDT_NODES__NDT_LOCALIZER_NODES_HPP_
