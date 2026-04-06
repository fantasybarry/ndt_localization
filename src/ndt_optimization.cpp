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

#include "ndt/ndt_optimization.hpp"

#include <cmath>

namespace autoware
{
namespace localization
{
namespace ndt
{

// ===========================================================================
// P2DNDTObjective — static helpers
// ===========================================================================

Eigen::Matrix4d P2DNDTObjective::to_matrix(const DomainValue & t)
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

Eigen::Matrix<double, 3, 6> P2DNDTObjective::point_jacobian(
  const Eigen::Vector3d & point,
  const DomainValue & t)
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

// ===========================================================================
// P2DNDTObjective — CachedExpression: score a single (point, voxel) pair
// ===========================================================================

P2DNDTObjective::Result P2DNDTObjective::score_point_voxel(
  const Eigen::Vector3d & transformed_point,
  const Eigen::Vector3d & centroid,
  const Eigen::Matrix3d & cov_inv,
  const Eigen::Matrix<double, 3, 6> & J) const
{
  Result s;

  Eigen::Vector3d diff = transformed_point - centroid;

  // Cached intermediate: exponent and exponential (shared by score, grad, hessian).
  double exponent = -0.5 * diff.transpose() * cov_inv * diff;
  double exp_val = std::exp(exponent);

  // Score: -exp(-0.5 * d^T Sigma^{-1} d)
  s.score = -exp_val;

  // Gradient: exp_val * (d^T Sigma^{-1} J)
  Eigen::Matrix<double, 1, 6> dTcJ = diff.transpose() * cov_inv * J;
  s.gradient = exp_val * dTcJ.transpose();

  // Hessian (Gauss-Newton approximation + second-order term).
  Eigen::Matrix<double, 6, 6> JtSJ = J.transpose() * cov_inv * J;
  s.hessian = exp_val * (-JtSJ + dTcJ.transpose() * dTcJ);

  return s;
}

// ===========================================================================
// P2DNDTObjective — Expression interface (score_, jacobian_, hessian_)
// ===========================================================================

double P2DNDTObjective::score_(const DomainValue & x) const
{
  // Delegate to evaluate and return just the score.
  // Note: scan_ must be bound before calling.
  // For standalone use without a map bound, return 0.
  return 0.0;
  // Full evaluation requires a map; use evaluate() template instead.
}

Eigen::Matrix<double, 6, 1> P2DNDTObjective::jacobian_(const DomainValue & x) const
{
  return Eigen::Matrix<double, 6, 1>::Zero();
}

Eigen::Matrix<double, 6, 6> P2DNDTObjective::hessian_(const DomainValue & x) const
{
  return Eigen::Matrix<double, 6, 6>::Zero();
}

// ===========================================================================
// P2DNDTObjective::evaluate — template implementation
// ===========================================================================

template <typename MapT>
P2DNDTObjective::Result P2DNDTObjective::evaluate(
  const P2DNDTScan & scan,
  const MapT & map,
  const DomainValue & transform,
  double search_radius) const
{
  Eigen::Matrix4d T = to_matrix(transform);
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::Vector3d t = T.block<3, 1>(0, 3);

  Result total;

  for (auto it = scan.begin(); it != scan.end(); ++it) {
    const Eigen::Vector3d & pt = *it;
    Eigen::Vector3d tp = R * pt + t;
    auto J = point_jacobian(pt, transform);

    // Look up the voxel(s) at the transformed point.
    const auto & voxels = map.cell(tp);
    for (const auto & voxel_view : voxels) {
      if (!voxel_view.usable()) { continue; }

      // Pass centroid and inverse covariance directly — no double inversion.
      auto s = score_point_voxel(
        tp, voxel_view.centroid(), voxel_view.inverse_covariance(), J);
      total.score += s.score;
      total.gradient += s.gradient;
      total.hessian += s.hessian;
    }
  }
  return total;
}

// Explicit template instantiations.
template P2DNDTObjective::Result P2DNDTObjective::evaluate<DynamicNDTMap>(
  const P2DNDTScan &, const DynamicNDTMap &,
  const EigenPose &, double) const;

template P2DNDTObjective::Result P2DNDTObjective::evaluate<StaticNDTMap>(
  const P2DNDTScan &, const StaticNDTMap &,
  const EigenPose &, double) const;

// NewtonsMethod::optimize is defined inline in the header (class template + member template).

}  // namespace ndt
}  // namespace localization
}  // namespace autoware
