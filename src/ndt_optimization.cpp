#include "ndt/ndt_optimization.hpp"

#include <cmath>

namespace ndt
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Eigen::Matrix4d P2DOptimizationProblem::to_matrix(const Transform6D & t)
{
  const double tx = t(0), ty = t(1), tz = t(2);
  const double roll = t(3), pitch = t(4), yaw = t(5);

  const double cr = std::cos(roll),  sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw),   sy = std::sin(yaw);

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  // ZYX rotation convention.
  T(0, 0) = cy * cp;
  T(0, 1) = cy * sp * sr - sy * cr;
  T(0, 2) = cy * sp * cr + sy * sr;
  T(1, 0) = sy * cp;
  T(1, 1) = sy * sp * sr + cy * cr;
  T(1, 2) = sy * sp * cr - cy * sr;
  T(2, 0) = -sp;
  T(2, 1) = cp * sr;
  T(2, 2) = cp * cr;

  T(0, 3) = tx;
  T(1, 3) = ty;
  T(2, 3) = tz;
  return T;
}

Eigen::Matrix<double, 3, 6> P2DOptimizationProblem::point_jacobian(
  const Eigen::Vector3d & point,
  const Transform6D & t)
{
  const double roll = t(3), pitch = t(4), yaw = t(5);
  const double cr = std::cos(roll),  sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw),   sy = std::sin(yaw);
  const double x = point.x(), y = point.y(), z = point.z();

  Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();

  // Derivatives w.r.t. translation are identity columns.
  J(0, 0) = 1.0;
  J(1, 1) = 1.0;
  J(2, 2) = 1.0;

  // d(Rp)/d(roll)
  J(0, 3) = y * (cy * sp * cr + sy * sr) + z * (-cy * sp * sr + sy * cr);
  J(1, 3) = y * (sy * sp * cr - cy * sr) + z * (-sy * sp * sr - cy * cr);
  J(2, 3) = y * cp * cr + z * (-cp * sr);

  // d(Rp)/d(pitch)
  J(0, 4) = x * (-cy * sp) + y * (cy * cp * sr) + z * (cy * cp * cr);
  J(1, 4) = x * (-sy * sp) + y * (sy * cp * sr) + z * (sy * cp * cr);
  J(2, 4) = x * (-cp) + y * (-sp * sr) + z * (-sp * cr);

  // d(Rp)/d(yaw)
  J(0, 5) = x * (-sy * cp) + y * (-sy * sp * sr - cy * cr) + z * (-sy * sp * cr + cy * sr);
  J(1, 5) = x * (cy * cp) + y * (cy * sp * sr - sy * cr) + z * (cy * sp * cr + sy * sr);
  J(2, 5) = 0.0;

  return J;
}

// ---------------------------------------------------------------------------
// CachedExpression pattern — score a single (point, voxel) pair
// ---------------------------------------------------------------------------

P2DScore P2DOptimizationProblem::score_point_voxel(
  const Eigen::Vector3d & transformed_point,
  const Voxel & voxel,
  const Eigen::Matrix<double, 3, 6> & J) const
{
  P2DScore s;

  Eigen::Vector3d diff = transformed_point - voxel.centroid;
  Eigen::Matrix3d cov_inv = voxel.covariance.inverse();

  // Cached intermediate: exponent and exponential (shared by score, grad, hessian).
  double exponent = -0.5 * diff.transpose() * cov_inv * diff;
  double exp_val = std::exp(exponent);

  // Score: −exp(−0.5 * d^T Σ^{-1} d)
  s.score = -exp_val;

  // Gradient: exp_val * (d^T Σ^{-1} J)
  Eigen::Matrix<double, 1, 6> dTcJ = diff.transpose() * cov_inv * J;
  s.gradient = exp_val * dTcJ.transpose();

  // Hessian (Gauss-Newton approximation + second-order term).
  Eigen::Matrix<double, 6, 6> JtSJ = J.transpose() * cov_inv * J;
  s.hessian = exp_val * (-JtSJ + dTcJ.transpose() * dTcJ);

  return s;
}

// ---------------------------------------------------------------------------
// Full evaluation over all scan points
// ---------------------------------------------------------------------------

template <typename MapT>
static P2DScore evaluate_impl(
  const P2DOptimizationProblem & problem,
  const P2DNDTScan & scan,
  const MapT & map,
  const P2DOptimizationProblem::Transform6D & transform,
  double search_radius)
{
  Eigen::Matrix4d T = P2DOptimizationProblem::to_matrix(transform);
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::Vector3d t = T.block<3, 1>(0, 3);

  P2DScore total;

  for (const auto & pt : scan) {
    Eigen::Vector3d tp = R * pt + t;
    auto J = P2DOptimizationProblem::point_jacobian(pt, transform);

    auto voxels = map.nearby_voxels(tp, search_radius);
    for (const auto * voxel : voxels) {
      auto s = problem.score_point_voxel(tp, *voxel, J);
      total.score += s.score;
      total.gradient += s.gradient;
      total.hessian += s.hessian;
    }
  }
  return total;
}

P2DScore P2DOptimizationProblem::evaluate(
  const P2DNDTScan & scan,
  const DynamicNDTMap & map,
  const Transform6D & transform,
  double search_radius) const
{
  return evaluate_impl(*this, scan, map, transform, search_radius);
}

P2DScore P2DOptimizationProblem::evaluate(
  const P2DNDTScan & scan,
  const StaticNDTMap & map,
  const Transform6D & transform,
  double search_radius) const
{
  return evaluate_impl(*this, scan, map, transform, search_radius);
}

// ---------------------------------------------------------------------------
// Newton Optimizer
// ---------------------------------------------------------------------------

NewtonOptimizer::NewtonOptimizer(const OptimizerParams & params)
: params_(params)
{
}

template <typename MapT>
NewtonOptimizer::Result NewtonOptimizer::optimize(
  const P2DNDTScan & scan,
  const MapT & map,
  const P2DOptimizationProblem::Transform6D & initial_estimate,
  const P2DOptimizationProblem & problem) const
{
  Result res;
  res.transform = initial_estimate;
  double prev_score = 0.0;

  for (int i = 0; i < params_.max_iterations; ++i) {
    auto eval = problem.evaluate(scan, map, res.transform);
    res.final_score = eval.score;
    res.iterations = i + 1;

    // Solve H * delta = -g  (Newton step).
    Eigen::Matrix<double, 6, 6> H = eval.hessian;
    H.diagonal().array() += 1e-6;  // Levenberg–Marquardt-style regularisation.

    Eigen::Matrix<double, 6, 1> delta = H.ldlt().solve(-eval.gradient);
    delta *= params_.step_size;

    res.transform += delta;

    // Check convergence.
    if (delta.norm() < params_.epsilon) {
      res.converged = true;
      break;
    }
    if (i > 0 && std::abs(eval.score - prev_score) < params_.score_epsilon) {
      res.converged = true;
      break;
    }
    prev_score = eval.score;
  }
  return res;
}

// Explicit template instantiations.
template NewtonOptimizer::Result NewtonOptimizer::optimize<DynamicNDTMap>(
  const P2DNDTScan &, const DynamicNDTMap &,
  const P2DOptimizationProblem::Transform6D &,
  const P2DOptimizationProblem &) const;

template NewtonOptimizer::Result NewtonOptimizer::optimize<StaticNDTMap>(
  const P2DNDTScan &, const StaticNDTMap &,
  const P2DOptimizationProblem::Transform6D &,
  const P2DOptimizationProblem &) const;

}  // namespace ndt
