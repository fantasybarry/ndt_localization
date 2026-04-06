#ifndef NDT__NDT_COMMON_HPP_
#define NDT__NDT_COMMON_HPP_

#include "common/types.hpp"
#include <ndt/visibility_control.hpp>

#include <Eigen/Core>

namespace autoware
{
namespace localization
{
namespace ndt
{

using autoware::common::types::float64_t;

/// The scalar type used for NDT computations.
using Real = float64_t;

/// 6-DoF pose vector: [tx, ty, tz, roll, pitch, yaw].
template<typename T>
using EigenPose = Eigen::Matrix<T, 6, 1>;

}  // namespace ndt
}  // namespace localization
}  // namespace autoware

#endif  // NDT__NDT_COMMON_HPP_
