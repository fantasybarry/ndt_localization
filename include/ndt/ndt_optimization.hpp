#ifndef NDT__NDT_OPTIMIZATION_HPP_
#define NDT__NDT_OPTIMIZATION_HPP_

#include "ndt/ndt_map.hpp"
#include "ndt/ndt_scan.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace ndt
{

/// Result of a single P2D NDT score evaluation for one scan point against one voxel.
/// All three quantities share intermediate terms, so they are computed together
/// via a CachedExpression approach (Magnusson 2009, Section 6.2).
struct P2DScore
{
  double score = 0.0;
  Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
};

/// Computes the P2D NDT cost function, gradient, and Hessian for a given
/// scan–map pair at a candidate transform.
///
/// The 6-DoF transform is parameterised as (tx, ty, tz, roll, pitch, yaw).
class P2DOptimizationProblem
{
public:
  using Transform6D = Eigen::Matrix<double, 6, 1>;

  P2DOptimizationProblem() = default;

  /// Evaluate the total score, gradient, and Hessian over all scan–voxel pairs.
  ///
  /// @param scan        Current LiDAR scan.
  /// @param map         NDT map (either Dynamic or Static — accessed through lookup).
  /// @param transform   Current 6-DoF pose estimate [tx, ty, tz, roll, pitch, yaw].
  /// @param search_radius  Radius for voxel neighbourhood lookup.
  /// @return Aggregated P2DScore.
  P2DScore evaluate(
    const P2DNDTScan & scan,
    const DynamicNDTMap & map,
    const Transform6D & transform,
    double search_radius = 2.0) const;

  /// Overload that accepts a StaticNDTMap.
  P2DScore evaluate(
    const P2DNDTScan & scan,
    const StaticNDTMap & map,
    const Transform6D & transform,
    double search_radius = 2.0) const;

private:
  /// Compute the score contribution from a single (point, voxel) pair.
  /// Uses the CachedExpression pattern: the exponential and its derivatives are
  /// computed once and shared between score, gradient, and Hessian.
  P2DScore score_point_voxel(
    const Eigen::Vector3d & transformed_point,
    const Voxel & voxel,
    const Eigen::Matrix<double, 3, 6> & jacobian_of_point) const;

  /// Build the 3×6 Jacobian of a transformed point w.r.t. the 6-DoF parameters.
  static Eigen::Matrix<double, 3, 6> point_jacobian(
    const Eigen::Vector3d & point,
    const Transform6D & transform);

  /// Convert a 6-DoF vector into a 4×4 homogeneous transform.
  static Eigen::Matrix4d to_matrix(const Transform6D & t);
};

// ---------------------------------------------------------------------------
// Newton-style optimiser
// ---------------------------------------------------------------------------

struct OptimizerParams
{
  int max_iterations = 30;
  double step_size = 1.0;
  double epsilon = 1e-4;       // convergence threshold on parameter change
  double score_epsilon = 1e-6; // convergence threshold on score change
};

/// A simple Newton / Gauss-Newton optimiser that uses the Hessian returned by
/// the P2D cost function to iteratively refine a 6-DoF pose.
class NewtonOptimizer
{
public:
  explicit NewtonOptimizer(const OptimizerParams & params = {});

  struct Result
  {
    P2DOptimizationProblem::Transform6D transform;
    double final_score = 0.0;
    int iterations = 0;
    bool converged = false;
  };

  /// Run optimisation.
  template <typename MapT>
  Result optimize(
    const P2DNDTScan & scan,
    const MapT & map,
    const P2DOptimizationProblem::Transform6D & initial_estimate,
    const P2DOptimizationProblem & problem) const;

private:
  OptimizerParams params_;
};

}  // namespace ndt

#endif  // NDT__NDT_OPTIMIZATION_HPP_
