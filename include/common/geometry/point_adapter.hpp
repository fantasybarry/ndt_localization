#ifndef COMMON__GEOMETRY__POINT_ADAPTER_HPP_
#define COMMON__GEOMETRY__POINT_ADAPTER_HPP_

#include "common/types.hpp"

namespace autoware
{
namespace common
{
namespace geometry
{
namespace point_adapter
{

using autoware::common::types::float32_t;

/// Primary templates for point access — specialise per point type.

template<typename PointT>
inline auto x_(const PointT & pt) { return static_cast<float32_t>(pt.x); }

template<typename PointT>
inline auto y_(const PointT & pt) { return static_cast<float32_t>(pt.y); }

template<typename PointT>
inline auto z_(const PointT & pt) { return static_cast<float32_t>(pt.z); }

template<typename PointT>
inline auto & xr_(PointT & pt) { return pt.x; }

template<typename PointT>
inline auto & yr_(PointT & pt) { return pt.y; }

template<typename PointT>
inline auto & zr_(PointT & pt) { return pt.z; }

}  // namespace point_adapter
}  // namespace geometry
}  // namespace common
}  // namespace autoware

#endif  // COMMON__GEOMETRY__POINT_ADAPTER_HPP_
