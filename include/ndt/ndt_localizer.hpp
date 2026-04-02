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
using EigenPose = Eigen::Matrix<double, 6, 1>;

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

  /// Register the current LiDAR scan.
  void register_scan(P2DNDTScan && scan) { scan_ = std::move(scan); }

  /// Run localisation from an initial estimate and return the refined pose.
  virtual PoseWithCovariance localize(
    const Eigen::Matrix<double, 6, 1> & initial_estimate)
  {
    if (!map_ || scan_.empty()) {
      return {};
    }

    auto result = optimizer_.optimize(scan_, *map_, initial_estimate, problem_);

    PoseWithCovariance output;
    output.pose = result.transform;
    output.score = result.final_score;
    output.valid = result.converged;

    // Approximate covariance from the inverse Hessian at the solution.
    auto final_eval = problem_.evaluate(scan_, *map_, result.transform);
    Eigen::Matrix<double, 6, 6> H = final_eval.hessian;
    // Regularise to guarantee invertibility.
    H.diagonal().array() += 1e-6;
    output.covariance = H.inverse();

    return output;
  }

protected:
  const MapT * map_ = nullptr;
  P2DNDTScan scan_;
  P2DOptimizationProblem problem_;
  NewtonOptimizer optimizer_;
};

// ---------------------------------------------------------------------------
// P2DNDTLocalizer — concrete implementation
// ---------------------------------------------------------------------------

/// Concrete Point-to-Distribution NDT localiser using a StaticNDTMap.
class P2DNDTLocalizer : public NDTLocalizerBase<StaticNDTMap>
{
public:
  explicit P2DNDTLocalizer(const OptimizerParams & params = {})
  : NDTLocalizerBase<StaticNDTMap>(params) {}
};

/// Concrete Point-to-Distribution NDT localiser using a DynamicNDTMap.
class P2DNDTDynamicLocalizer : public NDTLocalizerBase<DynamicNDTMap>
{
public:
  explicit P2DNDTDynamicLocalizer(const OptimizerParams & params = {})
  : NDTLocalizerBase<DynamicNDTMap>(params) {}
};

}  // namespace ndt

#endif  // NDT__NDT_LOCALIZER_HPP_
