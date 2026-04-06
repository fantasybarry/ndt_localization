#ifndef NDT__NDT_LOCALIZER_HPP_
#define NDT__NDT_LOCALIZER_HPP_

#include "ndt/ndt_map.hpp"
#include "ndt/ndt_optimization.hpp"
#include "ndt/ndt_scan.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <optional>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace autoware
{
namespace localization
{
namespace ndt
{
using CloudT = sensor_msgs::msg::PointCloud2;
using Transform = geometry_msgs::msg::TransformStamped;
using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
// EigenPose is defined in ndt_common.hpp (via ndt_map.hpp).

struct NDTLocalizerConfig
{
  int64_t guess_time_tolerance_ns = 100000000;
};

struct PoseWithCovariance
{
  Eigen::Matrix<double, 6, 1> pose;   // tx, ty, tz, roll, pitch, yaw
  Eigen::Matrix<double, 6, 6> covariance;
  double score = 0.0;
  bool valid = false;
};

// ---------------------------------------------------------------------------
// NDTLocalizerBase — template interface
// ---------------------------------------------------------------------------

/// Template base for NDT localisers.  MapT can be DynamicNDTMap or StaticNDTMap.
template <
  typename ScanT,
  typename MapT,
  typename NDTOptimizationProblemT,
  typename OptimizerT>
class NDTLocalizerBase
{
public:
  NDTLocalizerBase(
    const NDTLocalizerConfig & config,
    const NDTOptimizationProblemT & problem,
    const OptimizerT & optimizer,
    ScanT && scan,
    MapT && map)
  : config_{config},
    problem_{problem},
    optimizer_{optimizer},
    scan_{std::forward<ScanT>(scan)},
    map_{std::forward<MapT>(map)}{}
  

  virtual ~NDTLocalizerBase() = default;
  
  // -- Map management (matches Autoware set_map_impl / insert_to_map_impl) --
  void set_map(const CloudT & msg)
  {
    map_.clear();
    map_.insert(msg);
  }

  /// Incrementally add points to the existing map
  void insert_to_map(const CloudT & msg)
  {
    map_.insert(msg);
  }

  /// Register the reference map directly (non-ROS path)
  void register_map(const MapT & map) { map_ = map; }

  /// Register the current LiDAR scan.
  void register_scan(ScanT && scan) { scan_ = std::move(scan); }
  //void register_scan(P2DNDTScan && scan) { scan_ = std::move(scan); }
  

  // -- Core localisation --

  /// Run localisation from an initial Eigen pose and return the refined pose.
  /// Uses inverse-Hessian covariance approximation
  virtual PoseWithCovariance localize(
    const Eigen::Matrix<double, 6, 1> & initial_estimate)
  {
    if (scan_.empty()) {
      return {};
    }

    auto result = optimizer_.optimize(scan_, map_, initial_estimate, problem_);

    PoseWithCovariance output;
    output.pose = result.transform;
    output.score = result.final_score;
    output.valid = result.converged;

    // Approximate covariance from the inverse Hessian at the solution.
    auto final_eval = problem_.evaluate(scan_, map_, result.transform);
    Eigen::Matrix<double, 6, 6> H = final_eval.hessian;
    // Regularise to guarantee invertibility.
    H.diagonal().array() += 1e-6;
    output.covariance = H.inverse();

    // Also populate ROS covariance via virtual look
    set_covariance(problem_, initial_estimate, result.transform, output);

    return output;
  }

  /// ROS-message entry point: register a scan and produce a PoseWithCovarianceStamped.
  ///
  /// @throws std::logic_error if scan is older than the map
  /// @throws std::domain_error if initial guess timestamp is too far from scan.
  /// @throws std::runtime_error on optimisation failure.
  void register_measurement(
    const CloudT & msg,
    const Transform & transform_initial,
    PoseWithCovarianceStamped & pose_out)
  {
    validate_msg(msg);
    validate_guess(msg, transform_initial);

    // Convert ROS transform -> Eigen 6-DoF.
    EigenPose eig_initial = transform_to_pose(transform_initial);

    // Set the scan from the incoming message.
    scan_.clear();
    scan_.insert(msg);

    // Run localisation
    auto result = localize(eig_initial);

    if(!result.valid) {
      throw std::runtime_error(
        "NDT localizer has likely encountered a numerical "
        "error during optimization."
      );
    }

    // Convert back to ROS pose.
    pose_to_msg(result.pose, pose_out.pose.pose);
    pose_out.header.stamp = msg.header.stamp;
    pose_out.header.frame_id = map_frame_id();

    // Fill covariance array from Eigen matrix.
    for (int r = 0; r < 6; ++r){
      for (int c = 0; c < 6; ++c) {
        pose_out.pose.covariance[static_cast<size_t>(r * 6 + c)] = result.covariance(r, c);
      }
    }
  }

  // -- Accessors --
  const ScanT & scan() const noexcept { return scan_; }
  const MapT & map() const noexcept { return map_; }
  const OptimizerT & optimizer() const noexcept { return optimizer_; }
  const NDTLocalizerConfig & config() const noexcept { return config_; }
  const std::string & map_frame_id() const noexcept { return map_.frame_id(); }


protected:
  // -- Virtual hooks for subclass customisation --

  /// Override to provide a custom covariance computation
  /// Default: no-op (the base localize() already fills covariance via inverse Hessian).

  virtual void set_covariance(
    const NDTOptimizationProblemT & /*problem*/,
    const EigenPose & /*initial_guess*/,
    const EigenPose & /*pose_result*/,
    PoseWithCovariance & /*output*/) const 
  {
  }
  
  /// Validate that the scan timestamp is not older than the map
  virtual void validate_msg(const CloudT & msg) const{
    const auto msg_sec = msg.header.stamp.sec;
    const auto map_sec = map_.stamp_sec();
    if (msg_sec < map_sec){
      throw std::logic_error(
        "Lidar scan should not have a timestamp older than the "
        "timestamp of the current map.");
    }
  }

  /// Validate that the initial guess is within the time tolerance of the scan.
  virtual void validate_guess(
    const CloudT & msg,
    const Transform & transform_initial) const
  {
    const int64_t scan_ns = 
      static_cast<int64_t>(msg.header.stamp.sec) * 1000000000LL + msg.header.stamp.nanosec;
    const int64_t guess_ns = 
      static_cast<int64_t>(transform_initial.header.stamp.sec) * 1000000000LL + transform_initial.header.stamp.nanosec;

    if (std::abs(guess_ns - scan_ns) > config_.guess_time_tolerance_ns) {
      throw std::domain_error(
        "Initial guess is not within tolerance of the scan's timestamp. "
        "Either increase the tolerance or ensure timely initial pose guesses."
      );
    }
  }

  // -- Conversion helpers --

  static EigenPose transform_to_pose(const Transform & tf)
  {
    EigenPose p = EigenPose::Zero();
    p(0) = tf.transform.translation.x;
    p(1) = tf.transform.translation.y;
    p(2) = tf.transform.translation.z;

    const auto & q = tf.transform.rotation;
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

  static void pose_to_msg(
    const EigenPose & pose,
    geometry_msgs::msg::Pose & msg)
  {
    msg.position.x = pose(0);
    msg.position.y = pose(1);
    msg.position.z = pose(2);

    // Roll/pitch/yaw -> quaternion.
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

private:
  NDTLocalizerConfig config_;
  NDTOptimizationProblemT problem_;
  OptimizerT optimizer_;
  ScanT scan_;
  MapT map_;
};

// ---------------------------------------------------------------------------
// P2DNDTLocalizer — concrete Point-to-Distribution implementation
// ---------------------------------------------------------------------------

template <typename OptimizerT = NewtonOptimizer, typename MapT = StaticNDTMap>
class P2DNDTLocalizer : public NDTLocalizerBase<
    P2DNDTScan, MapT, P2DOptimizationProblem, OptimizerT>
{
public:
  using ParentT = NDTLocalizerBase<
    P2DNDTScan, MapT, P2DOptimizationProblem, OptimizerT>;
  P2DNDTLocalizer(
    const NDTLocalizerConfig & config,
    const OptimizerT & optimizer,
    P2DNDTScan && scan,
    MapT && map)
  : ParentT{
      config,
      P2DOptimizationProblem{},
      optimizer,
      std::forward<P2DNDTScan>(scan),
      std::forward<MapT>(map)} {}

protected:
  /// Override to compute covariance from inverse Hessian at the solution.
  void set_covariance(
    const P2DOptimizationProblem & problem,
    const EigenPose & /*initial_guess*/,
    const EigenPose & pose_result,
    PoseWithCovariance & output) const override
  {
    auto eval = problem.evaluate(this->scan(), this->map(), pose_result);
    Eigen::Matrix<double, 6, 6> H = eval.hessian;
    H.diagonal().array() += 1e-6;
    output.covariance = H.inverse();
  }  
};


} // namespace ndt
} // namespace localization
} // namespace autoware

#endif  // NDT__NDT_LOCALIZER_HPP_
