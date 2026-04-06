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

#include <ndt/ndt_localizer.hpp>
#include <ndt/ndt_map.hpp>
#include <ndt/ndt_scan.hpp>
#include <ndt/ndt_optimization.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int32.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>

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

/// Best-effort pose initializer: stores the latest initial pose and provides
/// it as the initial guess until replaced.
class BestEffortInitializer
{
public:
  using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;

  void set(const PoseWithCovarianceStamped & msg)
  {
    initial_pose_ = msg;
    received_ = true;
  }

  bool received() const { return received_; }
  const PoseWithCovarianceStamped & get() const { return initial_pose_; }

  void update(const PoseWithCovarianceStamped & pose) { initial_pose_ = pose; }

private:
  PoseWithCovarianceStamped initial_pose_;
  bool received_ = false;
};

/// P2D NDT Localizer ROS 2 Node (Autoware-compatible interface).
///
/// Input:
///   - ekf_pose_with_covariance           (PoseWithCovarianceStamped)  — initial pose
///   - pointcloud_map                     (PointCloud2, transient local) — map pointcloud
///   - points_raw                         (PointCloud2, sensor QoS)     — sensor pointcloud
///   - sensing/gnss/pose_with_covariance  (PoseWithCovarianceStamped)  — GNSS regularization
///
/// Output:
///   - ndt_pose                           (PoseStamped)                — estimated pose
///   - ndt_pose_with_covariance           (PoseWithCovarianceStamped)  — with covariance
///   - diagnostics                        (DiagnosticArray)            — diagnostics
///   - points_aligned                     (PointCloud2)                — [debug] aligned scan
///   - points_aligned_no_ground           (PointCloud2)                — [debug] no ground aligned
///   - initial_pose_with_covariance       (PoseWithCovarianceStamped)  — [debug] initial pose used
///   - multi_ndt_pose                     (PoseArray)                  — [debug] multi-NDT poses
///   - multi_initial_pose                 (PoseArray)                  — [debug] multi initial poses
///   - exe_time_ms                        (Float32)                    — [debug] execution time
///   - transform_probability              (Float64)                    — [debug] score
///   - no_ground_transform_probability    (Float64)                    — [debug] no-ground score
///   - iteration_num                      (Int32)                      — [debug] iterations
///   - initial_to_result_distance         (Float32)                    — [debug] init→result dist
///   - initial_to_result_distance_old     (Float32)                    — [debug] distance (prev)
///   - initial_to_result_distance_old_2d  (Float32)                    — [debug] 2D distance
///   - initial_to_result_relative_pose    (PoseStamped)                — [debug] relative pose
///   - nearest_voxel_transformation_likelihood        (Float64)        — [debug] NVTL score
///   - no_ground_nearest_voxel_transformation_likelihood (Float64)     — [debug] no-ground NVTL
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
  using Transform = geometry_msgs::msg::TransformStamped;

  static constexpr auto EPS = std::numeric_limits<ndt::Real>::epsilon();

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
    this->declare_parameter<int>("scan_capacity", 55000);
    this->declare_parameter<double>("search_radius", 2.0);
    this->declare_parameter<double>("regularization_scale", 0.01);

    // Voxel grid config for the NDT map.
    this->declare_parameter<double>("map.min_x", -500.0);
    this->declare_parameter<double>("map.min_y", -500.0);
    this->declare_parameter<double>("map.min_z", -10.0);
    this->declare_parameter<double>("map.max_x", 500.0);
    this->declare_parameter<double>("map.max_y", 500.0);
    this->declare_parameter<double>("map.max_z", 50.0);
    this->declare_parameter<double>("map.voxel_x", 1.0);
    this->declare_parameter<double>("map.voxel_y", 1.0);
    this->declare_parameter<double>("map.voxel_z", 1.0);
    this->declare_parameter<int>("map.capacity", 1000000);

    map_frame_ = this->get_parameter("map_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    score_threshold_ = this->get_parameter("score_threshold").as_double();
    scan_capacity_ = static_cast<std::size_t>(this->get_parameter("scan_capacity").as_int());
    search_radius_ = this->get_parameter("search_radius").as_double();
    regularization_scale_ = this->get_parameter("regularization_scale").as_double();

    // Build optimizer.
    ndt::NewtonsMethodParams opt_params;
    opt_params.max_iterations = this->get_parameter("max_iterations").as_int();
    opt_params.step_size = this->get_parameter("step_size").as_double();
    opt_params.epsilon = this->get_parameter("epsilon").as_double();
    optimizer_ = std::make_unique<ndt::NewtonOptimizer>(opt_params);

    // Build map config.
    Eigen::Vector3d min_pt(
      this->get_parameter("map.min_x").as_double(),
      this->get_parameter("map.min_y").as_double(),
      this->get_parameter("map.min_z").as_double());
    Eigen::Vector3d max_pt(
      this->get_parameter("map.max_x").as_double(),
      this->get_parameter("map.max_y").as_double(),
      this->get_parameter("map.max_z").as_double());
    Eigen::Vector3d voxel_size(
      this->get_parameter("map.voxel_x").as_double(),
      this->get_parameter("map.voxel_y").as_double(),
      this->get_parameter("map.voxel_z").as_double());
    auto capacity = static_cast<std::size_t>(this->get_parameter("map.capacity").as_int());

    map_config_ = std::make_shared<autoware::perception::filters::voxel_grid::Config>(
      min_pt, max_pt, voxel_size, capacity);

    // -- Publishers (output) --
    ndt_pose_pub_ = this->create_publisher<PoseStamped>(
      "ndt_pose", rclcpp::QoS{10});
    ndt_pose_with_cov_pub_ = this->create_publisher<PoseWithCovarianceStamped>(
      "ndt_pose_with_covariance", rclcpp::QoS{10});
    diagnostics_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "diagnostics", rclcpp::QoS{10});
    points_aligned_pub_ = this->create_publisher<CloudT>(
      "points_aligned", rclcpp::QoS{10});
    points_aligned_no_ground_pub_ = this->create_publisher<CloudT>(
      "points_aligned_no_ground", rclcpp::QoS{10});
    initial_pose_with_cov_pub_ = this->create_publisher<PoseWithCovarianceStamped>(
      "initial_pose_with_covariance", rclcpp::QoS{10});
    multi_ndt_pose_pub_ = this->create_publisher<PoseArray>(
      "multi_ndt_pose", rclcpp::QoS{10});
    multi_initial_pose_pub_ = this->create_publisher<PoseArray>(
      "multi_initial_pose", rclcpp::QoS{10});
    exe_time_ms_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "exe_time_ms", rclcpp::QoS{10});
    transform_probability_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "transform_probability", rclcpp::QoS{10});
    no_ground_transform_probability_pub_ = this->create_publisher<std_msgs::msg::Float64>(
      "no_ground_transform_probability", rclcpp::QoS{10});
    iteration_num_pub_ = this->create_publisher<std_msgs::msg::Int32>(
      "iteration_num", rclcpp::QoS{10});
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

    RCLCPP_INFO(this->get_logger(), "P2DNDTLocalizerNode initialized");
  }

private:
  // ---- Callbacks ----

  void on_map(const CloudT & msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received map with %u points", msg.width);
    map_ = std::make_unique<ndt::StaticNDTMap>(*map_config_);
    map_->insert(msg);
    map_received_ = true;
  }

  void on_initial_pose(const PoseWithCovarianceStamped & msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received initial pose (EKF)");
    pose_initializer_.set(msg);
  }

  void on_gnss_pose(const PoseWithCovarianceStamped & msg)
  {
    gnss_pose_ = msg;
    gnss_received_ = true;
  }

  void on_scan(const CloudT & msg)
  {
    auto t_start = std::chrono::steady_clock::now();

    if (!map_received_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "No map received yet, skipping scan");
      return;
    }
    if (!pose_initializer_.received()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "No initial pose received yet, skipping scan");
      return;
    }

    // Build scan.
    ndt::P2DNDTScan scan(msg, scan_capacity_);

    // Get initial guess.
    auto guess_msg = pose_initializer_.get();
    ndt::EigenPose initial_guess = pose_msg_to_eigen(guess_msg);

    // Publish the initial pose used (debug).
    initial_pose_with_cov_pub_->publish(guess_msg);

    // Run NDT optimization.
    ndt::P2DNDTOptimizationProblem problem;
    auto result = optimizer_->optimize(scan, *map_, initial_guess, problem);

    auto t_end = std::chrono::steady_clock::now();
    double exe_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (!result.converged) {
      RCLCPP_WARN(this->get_logger(), "NDT did not converge");
      publish_diagnostics("NDT did not converge", false);
      return;
    }

    // Compute covariance from inverse Hessian.
    auto eval = problem.evaluate(scan, *map_, result.transform, search_radius_);
    Eigen::Matrix<double, 6, 6> H = eval.hessian;
    H.diagonal().array() += 1e-6;
    Eigen::Matrix<double, 6, 6> cov = H.inverse();

    // ---- Publish ndt_pose (PoseStamped) ----
    PoseStamped pose_stamped;
    pose_stamped.header.stamp = msg.header.stamp;
    pose_stamped.header.frame_id = map_frame_;
    eigen_to_pose_msg(result.transform, pose_stamped.pose);
    ndt_pose_pub_->publish(pose_stamped);

    // ---- Publish ndt_pose_with_covariance ----
    PoseWithCovarianceStamped pose_cov;
    pose_cov.header = pose_stamped.header;
    pose_cov.pose.pose = pose_stamped.pose;
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        pose_cov.pose.covariance[static_cast<size_t>(r * 6 + c)] = cov(r, c);
      }
    }
    ndt_pose_with_cov_pub_->publish(pose_cov);

    // ---- Publish TF: map → base_link ----
    Transform tf;
    tf.header = pose_stamped.header;
    tf.child_frame_id = base_frame_;
    tf.transform.translation.x = pose_stamped.pose.position.x;
    tf.transform.translation.y = pose_stamped.pose.position.y;
    tf.transform.translation.z = pose_stamped.pose.position.z;
    tf.transform.rotation = pose_stamped.pose.orientation;
    tf_broadcaster_.sendTransform(tf);

    // ---- Debug topics ----

    // exe_time_ms
    std_msgs::msg::Float32 exe_msg;
    exe_msg.data = static_cast<float>(exe_time);
    exe_time_ms_pub_->publish(exe_msg);

    // transform_probability (NDT score)
    std_msgs::msg::Float64 score_msg;
    score_msg.data = result.final_score;
    transform_probability_pub_->publish(score_msg);
    // no_ground version uses same score (no ground filtering implemented)
    no_ground_transform_probability_pub_->publish(score_msg);

    // nearest_voxel_transformation_likelihood
    nvtl_pub_->publish(score_msg);
    no_ground_nvtl_pub_->publish(score_msg);

    // iteration_num
    std_msgs::msg::Int32 iter_msg;
    iter_msg.data = result.iterations;
    iteration_num_pub_->publish(iter_msg);

    // initial_to_result_distance
    double dx = result.transform(0) - initial_guess(0);
    double dy = result.transform(1) - initial_guess(1);
    double dz = result.transform(2) - initial_guess(2);
    double dist_3d = std::sqrt(dx * dx + dy * dy + dz * dz);
    double dist_2d = std::sqrt(dx * dx + dy * dy);

    std_msgs::msg::Float32 dist_msg;
    dist_msg.data = static_cast<float>(dist_3d);
    initial_to_result_distance_pub_->publish(dist_msg);
    initial_to_result_distance_old_pub_->publish(dist_msg);

    std_msgs::msg::Float32 dist_2d_msg;
    dist_2d_msg.data = static_cast<float>(dist_2d);
    initial_to_result_distance_old_2d_pub_->publish(dist_2d_msg);

    // initial_to_result_relative_pose
    PoseStamped rel_pose;
    rel_pose.header = pose_stamped.header;
    ndt::EigenPose rel = result.transform - initial_guess;
    eigen_to_pose_msg(rel, rel_pose.pose);
    initial_to_result_relative_pose_pub_->publish(rel_pose);

    // diagnostics
    publish_diagnostics("OK", true);

    // Update initializer for next iteration.
    pose_initializer_.update(pose_cov);
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

  static ndt::EigenPose pose_msg_to_eigen(const PoseWithCovarianceStamped & msg)
  {
    ndt::EigenPose p = ndt::EigenPose::Zero();
    p(0) = msg.pose.pose.position.x;
    p(1) = msg.pose.pose.position.y;
    p(2) = msg.pose.pose.position.z;

    const auto & q = msg.pose.pose.orientation;
    double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    p(3) = std::atan2(sinr_cosp, cosr_cosp);

    double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    p(4) = (std::abs(sinp) >= 1.0)
      ? std::copysign(M_PI / 2.0, sinp)
      : std::asin(sinp);

    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    p(5) = std::atan2(siny_cosp, cosy_cosp);

    return p;
  }

  static void eigen_to_pose_msg(
    const ndt::EigenPose & pose,
    geometry_msgs::msg::Pose & msg)
  {
    msg.position.x = pose(0);
    msg.position.y = pose(1);
    msg.position.z = pose(2);

    const double cr = std::cos(pose(3) * 0.5);
    const double sr = std::sin(pose(3) * 0.5);
    const double cp = std::cos(pose(4) * 0.5);
    const double sp = std::sin(pose(4) * 0.5);
    const double cy = std::cos(pose(5) * 0.5);
    const double sy = std::sin(pose(5) * 0.5);

    msg.orientation.w = cr * cp * cy + sr * sp * sy;
    msg.orientation.x = sr * cp * cy - cr * sp * sy;
    msg.orientation.y = cr * sp * cy + sr * cp * sy;
    msg.orientation.z = cr * cp * sy - sr * sp * cy;
  }

  // ---- Members ----

  // Core NDT components.
  BestEffortInitializer pose_initializer_;
  std::unique_ptr<ndt::NewtonOptimizer> optimizer_;
  std::unique_ptr<ndt::StaticNDTMap> map_;
  std::shared_ptr<autoware::perception::filters::voxel_grid::Config> map_config_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  // GNSS regularization.
  PoseWithCovarianceStamped gnss_pose_;
  bool gnss_received_ = false;
  double regularization_scale_ = 0.01;

  // Subscribers (input).
  rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr ekf_pose_sub_;
  rclcpp::Subscription<CloudT>::SharedPtr map_sub_;
  rclcpp::Subscription<CloudT>::SharedPtr scan_sub_;
  rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr gnss_pose_sub_;

  // Publishers (output).
  rclcpp::Publisher<PoseStamped>::SharedPtr ndt_pose_pub_;
  rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr ndt_pose_with_cov_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::Publisher<CloudT>::SharedPtr points_aligned_pub_;
  rclcpp::Publisher<CloudT>::SharedPtr points_aligned_no_ground_pub_;
  rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr initial_pose_with_cov_pub_;
  rclcpp::Publisher<PoseArray>::SharedPtr multi_ndt_pose_pub_;
  rclcpp::Publisher<PoseArray>::SharedPtr multi_initial_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr exe_time_ms_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr transform_probability_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr no_ground_transform_probability_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr iteration_num_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr initial_to_result_distance_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr initial_to_result_distance_old_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr initial_to_result_distance_old_2d_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr initial_to_result_relative_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr nvtl_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr no_ground_nvtl_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<PoseArray>::SharedPtr monte_carlo_initial_pose_pub_;

  // Config.
  std::string map_frame_;
  std::string base_frame_;
  bool map_received_ = false;
  double score_threshold_ = 2.0;
  std::size_t scan_capacity_ = 55000;
  double search_radius_ = 2.0;
};

}  // namespace ndt_nodes
}  // namespace localization
}  // namespace autoware

#endif  // NDT_NODES__NDT_LOCALIZER_NODES_HPP_
