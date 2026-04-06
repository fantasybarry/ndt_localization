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

#include <localization_common/localizer_base.hpp>
#include <ndt/ndt_localizer.hpp>
#include <ndt/ndt_map.hpp>
#include <ndt/ndt_scan.hpp>
#include <ndt/ndt_optimization.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <string>
#include <memory>
#include <limits>
#include <cmath>

namespace autoware
{
namespace localization
{
namespace ndt_nodes
{

/// Best-effort pose initializer: stores the latest /initialpose and provides
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

  /// Update the stored pose with the latest localisation result so the next
  /// scan uses the previous output as the initial guess.
  void update(const PoseWithCovarianceStamped & pose) { initial_pose_ = pose; }

private:
  PoseWithCovarianceStamped initial_pose_;
  bool received_ = false;
};

/// P2D NDT Localizer ROS 2 Node.
///
/// Subscribes to:
///   - ndt_map        (PointCloud2, transient local)  — the NDT reference map
///   - points_in      (PointCloud2, sensor QoS)       — incoming LiDAR scans
///   - /localization/initialpose (PoseWithCovarianceStamped) — initial guess
///
/// Publishes:
///   - ndt_pose       (PoseWithCovarianceStamped)     — localised pose
///   - map → base_link TF
///
/// Template parameters:
///   OptimizerT         — optimizer type (default: NewtonOptimizer)
///   PoseInitializerT   — initial pose strategy (default: BestEffortInitializer)
template <
  typename OptimizerT = ndt::NewtonOptimizer,
  typename PoseInitializerT = BestEffortInitializer>
class P2DNDTLocalizerNode
  : public localization_common::RelativeLocalizerNode<
      sensor_msgs::msg::PointCloud2,
      sensor_msgs::msg::PointCloud2,
      ndt::P2DNDTLocalizer,
      PoseInitializerT>
{
public:
  using CloudT = sensor_msgs::msg::PointCloud2;
  using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
  using Transform = geometry_msgs::msg::TransformStamped;
  using ParentT = localization_common::RelativeLocalizerNode<
    CloudT, CloudT, ndt::P2DNDTLocalizer, PoseInitializerT>;
  using EigTranslation = Eigen::Vector3d;
  using EigRotation = Eigen::Quaterniond;

  static constexpr auto EPS = std::numeric_limits<ndt::Real>::epsilon();

  P2DNDTLocalizerNode(
    const std::string & node_name,
    const std::string & name_space,
    PoseInitializerT pose_initializer = PoseInitializerT{})
  : ParentT(node_name, name_space),
    pose_initializer_(std::move(pose_initializer))
  {
    // Declare localizer-specific parameters.
    this->declare_parameter<int>("max_iterations", 30);
    this->declare_parameter<double>("step_size", 1.0);
    this->declare_parameter<double>("epsilon", 1e-4);
    this->declare_parameter<double>("score_threshold", 2.0);
    this->declare_parameter<int>("scan_capacity", 55000);
    this->declare_parameter<double>("search_radius", 2.0);

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

    // Build optimizer.
    ndt::NewtonsMethodParams opt_params;
    opt_params.max_iterations = this->get_parameter("max_iterations").as_int();
    opt_params.step_size = this->get_parameter("step_size").as_double();
    opt_params.epsilon = this->get_parameter("epsilon").as_double();
    optimizer_ = std::make_unique<OptimizerT>(opt_params);

    score_threshold_ = this->get_parameter("score_threshold").as_double();
    scan_capacity_ = static_cast<std::size_t>(this->get_parameter("scan_capacity").as_int());
    search_radius_ = this->get_parameter("search_radius").as_double();

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

    RCLCPP_INFO(this->get_logger(), "P2DNDTLocalizerNode initialized");
  }

protected:
  /// Handle incoming NDT map.
  void on_map(const CloudT & msg) override
  {
    RCLCPP_INFO(this->get_logger(), "Received NDT map with %u points", msg.width);
    map_ = std::make_unique<ndt::StaticNDTMap>(*map_config_);
    map_->insert(msg);
    map_received_ = true;
  }

  /// Handle incoming LiDAR scan — run localisation.
  void on_scan(const CloudT & msg) override
  {
    if (!map_received_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "No NDT map received yet, skipping scan");
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

    // Get initial guess from the pose initializer.
    auto guess_msg = pose_initializer_.get();
    ndt::EigenPose initial_guess = pose_msg_to_eigen(guess_msg);

    // Run localisation.
    ndt::P2DNDTOptimizationProblem problem;
    auto result = optimizer_->optimize(
      scan, *map_, initial_guess, problem);

    if (!result.converged) {
      RCLCPP_WARN(this->get_logger(), "NDT did not converge");
      return;
    }

    // Build output pose message.
    PoseWithCovarianceStamped pose_out;
    pose_out.header.stamp = msg.header.stamp;
    pose_out.header.frame_id = this->map_frame();
    eigen_to_pose_msg(result.transform, pose_out.pose.pose);

    // Compute covariance from inverse Hessian.
    auto eval = problem.evaluate(scan, *map_, result.transform, search_radius_);
    Eigen::Matrix<double, 6, 6> H = eval.hessian;
    H.diagonal().array() += 1e-6;
    Eigen::Matrix<double, 6, 6> cov = H.inverse();
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        pose_out.pose.covariance[static_cast<size_t>(r * 6 + c)] = cov(r, c);
      }
    }

    // Validate.
    localization_common::OptimizedRegistrationSummary summary;
    summary.score = result.final_score;
    summary.iterations = result.iterations;
    summary.converged = result.converged;

    if (!validate_output(summary, pose_out)) {
      RCLCPP_WARN(this->get_logger(), "Output validation failed (score: %.4f)", summary.score);
      return;
    }

    // Publish and update initializer for next iteration.
    this->publish_pose(pose_out);
    pose_initializer_.update(pose_out);
  }

  /// Handle initial pose.
  void on_initial_pose(const PoseWithCovarianceStamped & msg) override
  {
    RCLCPP_INFO(this->get_logger(), "Received initial pose");
    pose_initializer_.set(msg);
  }

  /// Validate the registration result against the score threshold.
  bool validate_output(
    const localization_common::OptimizedRegistrationSummary & summary,
    const PoseWithCovarianceStamped & /*pose*/) override
  {
    return summary.converged && (std::abs(summary.score) > EPS);
  }

private:
  // -- Eigen ↔ ROS conversion helpers --

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

  // Members.
  PoseInitializerT pose_initializer_;
  std::unique_ptr<OptimizerT> optimizer_;
  std::unique_ptr<ndt::StaticNDTMap> map_;
  std::shared_ptr<autoware::perception::filters::voxel_grid::Config> map_config_;
  bool map_received_ = false;
  double score_threshold_ = 2.0;
  std::size_t scan_capacity_ = 55000;
  double search_radius_ = 2.0;
};

}  // namespace ndt_nodes
}  // namespace localization
}  // namespace autoware

#endif  // NDT_NODES__NDT_LOCALIZER_NODES_HPP_
