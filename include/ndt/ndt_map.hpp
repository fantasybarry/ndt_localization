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

#ifndef NDT__NDT_MAP_HPP_
#define NDT__NDT_MAP_HPP_

#include <ndt/ndt_common.hpp>
#include <ndt/ndt_voxel.hpp>
#include <ndt/ndt_voxel_view.hpp>
#include <helper_functions/crtp.hpp>
#include <autoware/perception/filters/voxel_grid/config.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <time_utils/time_utils.hpp>
#include <limits>
#include <unordered_map>
#include <utility>
#include <string>
#include <vector>
#include "common/types.hpp"
#include "common/geometry/point_adapter.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using autoware::common::types::float32_t;

namespace autoware
{
namespace localization
{
namespace ndt
{

/// Validate that a PointCloud2 message has the required NDT map fields.
uint32_t NDT_PUBLIC validate_pcl_map(const sensor_msgs::msg::PointCloud2 & msg);

// ---------------------------------------------------------------------------
// Voxel — standalone struct (kept from our implementation)
// ---------------------------------------------------------------------------

/// A single NDT voxel storing a centroid and covariance computed from contained points.
struct Voxel
{
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  int64_t point_count = 0;

  /// Whether the voxel has enough points to be usable (minimum 3 for a 3D covariance).
  bool usable() const { return point_count >= 3; }
};

// ---------------------------------------------------------------------------
// NDTMapBase — CRTP base class (aligned with Autoware.Auto)
// ---------------------------------------------------------------------------

template<typename Derived, typename VoxelT>
class NDTMapBase : public common::helper_functions::crtp<Derived>
{
public:
  using Grid = std::unordered_map<uint64_t, VoxelT>;
  using Point = Eigen::Vector3d;
  using Config = autoware::perception::filters::voxel_grid::Config;
  using TimePoint = std::chrono::system_clock::time_point;
  using VoxelViewVector = std::vector<VoxelView<VoxelT>>;

  explicit NDTMapBase(const Config & voxel_grid_config)
  : m_config(voxel_grid_config), m_map(m_config.get_capacity())
  {
    m_output_vector.reserve(1U);
  }

  // Maps should be moved rather than being copied.
  NDTMapBase(const NDTMapBase &) = delete;
  NDTMapBase & operator=(const NDTMapBase &) = delete;
  NDTMapBase(NDTMapBase &&) = default;
  NDTMapBase & operator=(NDTMapBase &&) = default;

  /// Look up the voxel cell at a given (x, y, z) coordinate.
  const VoxelViewVector & cell(float32_t x, float32_t y, float32_t z) const
  {
    return cell(Point({x, y, z}));
  }

  /// Look up the voxel cell at a given point.
  const VoxelViewVector & cell(const Point & pt) const
  {
    m_output_vector.clear();
    const auto vx_it = m_map.find(m_config.index(pt));
    if (vx_it != m_map.end() && vx_it->second.usable()) {
      m_output_vector.emplace_back(vx_it->second);
    }
    return m_output_vector;
  }

  /// Insert a PointCloud2 message into the map (delegates to Derived::insert_).
  void insert(const sensor_msgs::msg::PointCloud2 & msg)
  {
    m_stamp = ::time_utils::from_message(msg.header.stamp);
    m_frame_id = msg.header.frame_id;
    this->impl().insert_(msg);
  }

  /// Insert a PCL point cloud (kept from our implementation for non-ROS paths).
  void insert(const pcl::PointCloud<pcl::PointXYZ> & cloud)
  {
    this->impl().insert_pcl_(cloud);
  }

  uint64_t size() const noexcept { return m_map.size(); }

  auto cell_size() const noexcept { return m_config.get_voxel_size(); }

  // -- Iterators --
  typename Grid::const_iterator begin() const noexcept { return cbegin(); }
  typename Grid::iterator begin() noexcept { return m_map.begin(); }
  typename Grid::const_iterator cbegin() const noexcept { return m_map.cbegin(); }
  typename Grid::const_iterator end() const noexcept { return cend(); }
  typename Grid::iterator end() noexcept { return m_map.end(); }
  typename Grid::const_iterator cend() const noexcept { return m_map.cend(); }

  void clear() noexcept { m_map.clear(); }

  TimePoint stamp() const noexcept { return m_stamp; }

  /// Return the seconds component of the stamp for simple comparisons.
  int32_t stamp_sec() const noexcept
  {
    auto epoch = m_stamp.time_since_epoch();
    return static_cast<int32_t>(
      std::chrono::duration_cast<std::chrono::seconds>(epoch).count());
  }

  const std::string & frame_id() const noexcept { return m_frame_id; }

protected:
  auto index(const Point & pt) const { return m_config.index(pt); }

  VoxelT & voxel(uint64_t idx) { return m_map[idx]; }

  auto emplace(uint64_t key, const VoxelT && vx)
  {
    return m_map.emplace(key, std::move(vx));
  }

private:
  mutable VoxelViewVector m_output_vector;
  const Config m_config;
  Grid m_map;
  TimePoint m_stamp{};
  std::string m_frame_id{};
};

// ---------------------------------------------------------------------------
// DynamicNDTMap — builds NDT on-the-fly using Welford's online algorithm
// ---------------------------------------------------------------------------

class NDT_PUBLIC DynamicNDTMap
  : public NDTMapBase<DynamicNDTMap, DynamicNDTVoxel>
{
public:
  using Voxel = DynamicNDTVoxel;
  using Grid = std::unordered_map<uint64_t, Voxel>;
  using Config = autoware::perception::filters::voxel_grid::Config;
  using Point = Eigen::Vector3d;

  using NDTMapBase::NDTMapBase;

  /// Insert from a ROS PointCloud2 message.
  void insert_(const sensor_msgs::msg::PointCloud2 & msg)
  {
    sensor_msgs::PointCloud2ConstIterator<float32_t> x_it(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float32_t> y_it(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float32_t> z_it(msg, "z");

    while (x_it != x_it.end() &&
      y_it != y_it.end() &&
      z_it != z_it.end())
    {
      const auto pt = Point({*x_it, *y_it, *z_it});
      const auto voxel_idx = index(pt);
      voxel(voxel_idx).add_observation(pt);

      ++x_it;
      ++y_it;
      ++z_it;
    }
    // Stabilise covariance after inserting all points.
    for (auto & vx_it : *this) {
      auto & vx = vx_it.second;
      (void) vx.try_stabilize();
    }
  }

  /// Insert from a PCL point cloud (Welford's algorithm, kept from our implementation).
  void insert_pcl_(const pcl::PointCloud<pcl::PointXYZ> & cloud)
  {
    for (const auto & pcl_pt : cloud.points) {
      const auto pt = Point({pcl_pt.x, pcl_pt.y, pcl_pt.z});
      const auto voxel_idx = index(pt);
      voxel(voxel_idx).add_observation(pt);
    }
    for (auto & vx_it : *this) {
      auto & vx = vx_it.second;
      (void) vx.try_stabilize();
    }
  }
};

// ---------------------------------------------------------------------------
// StaticNDTMap — loads pre-computed inverse covariance from PointCloud2
// ---------------------------------------------------------------------------

class NDT_PUBLIC StaticNDTMap
  : public NDTMapBase<StaticNDTMap, StaticNDTVoxel>
{
public:
  using NDTMapBase::NDTMapBase;
  using Voxel = StaticNDTVoxel;

  /// Insert from a ROS PointCloud2 containing pre-computed NDT voxels.
  /// Expected fields: x, y, z, icov_xx, icov_xy, icov_xz, icov_yy, icov_yz, icov_zz, cell_id.
  void insert_(const sensor_msgs::msg::PointCloud2 & msg)
  {
    if (validate_pcl_map(msg) == 0U) {
      throw std::runtime_error(
        "Point cloud representing the ndt map is either empty "
        "or does not have the correct format.");
    }

    sensor_msgs::PointCloud2ConstIterator<Real> x_it(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<Real> y_it(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<Real> z_it(msg, "z");
    sensor_msgs::PointCloud2ConstIterator<Real> icov_xx_it(msg, "icov_xx");
    sensor_msgs::PointCloud2ConstIterator<Real> icov_xy_it(msg, "icov_xy");
    sensor_msgs::PointCloud2ConstIterator<Real> icov_xz_it(msg, "icov_xz");
    sensor_msgs::PointCloud2ConstIterator<Real> icov_yy_it(msg, "icov_yy");
    sensor_msgs::PointCloud2ConstIterator<Real> icov_yz_it(msg, "icov_yz");
    sensor_msgs::PointCloud2ConstIterator<Real> icov_zz_it(msg, "icov_zz");
    sensor_msgs::PointCloud2ConstIterator<uint32_t> cell_id_it(msg, "cell_id");

    while (x_it != x_it.end() &&
      y_it != y_it.end() &&
      z_it != z_it.end() &&
      icov_xx_it != icov_xx_it.end() &&
      icov_xy_it != icov_xy_it.end() &&
      icov_xz_it != icov_xz_it.end() &&
      icov_yy_it != icov_yy_it.end() &&
      icov_yz_it != icov_yz_it.end() &&
      icov_zz_it != icov_zz_it.end() &&
      cell_id_it != cell_id_it.end())
    {
      const Point centroid{*x_it, *y_it, *z_it};
      const auto voxel_idx = index(centroid);

      // cell_id is stored as two 32-bit ints representing a 64-bit key.
      Grid::key_type received_idx = 0U;
      std::memcpy(&received_idx, &cell_id_it[0U], sizeof(received_idx));

      if (voxel_idx != received_idx) {
        throw std::domain_error(
          "NDTVoxelMap: Pointcloud representing the ndt map "
          "does not have a matching grid configuration with "
          "the map representation it is being inserted to. The cell IDs do not match.");
      }

      Eigen::Matrix3d inv_covariance;
      inv_covariance << *icov_xx_it, *icov_xy_it, *icov_xz_it,
        *icov_xy_it, *icov_yy_it, *icov_yz_it,
        *icov_xz_it, *icov_yz_it, *icov_zz_it;
      const Voxel vx{centroid, inv_covariance};

      const auto insert_res = emplace(voxel_idx, Voxel{centroid, inv_covariance});
      if (!insert_res.second) {
        insert_res.first->second = vx;
      }

      ++x_it;
      ++y_it;
      ++z_it;
      ++icov_xx_it;
      ++icov_xy_it;
      ++icov_xz_it;
      ++icov_yy_it;
      ++icov_yz_it;
      ++icov_zz_it;
      ++cell_id_it;
    }
  }

  /// Insert from a PCD file (kept from our implementation for non-ROS paths).
  void insert_pcl_(const pcl::PointCloud<pcl::PointXYZ> & /*cloud*/)
  {
    // StaticNDTMap expects pre-computed voxels via PointCloud2.
    // Use DynamicNDTMap for raw point cloud insertion.
  }
};

}  // namespace ndt
}  // namespace localization

// ---------------------------------------------------------------------------
// Point adapter specialisations for Eigen::Vector3d
// ---------------------------------------------------------------------------

namespace common
{
namespace geometry
{
namespace point_adapter
{

template<>
inline NDT_PUBLIC auto x_(const Eigen::Vector3d & pt)
{
  return static_cast<float32_t>(pt(0));
}

template<>
inline NDT_PUBLIC auto y_(const Eigen::Vector3d & pt)
{
  return static_cast<float32_t>(pt(1));
}

template<>
inline NDT_PUBLIC auto z_(const Eigen::Vector3d & pt)
{
  return static_cast<float32_t>(pt(2));
}

template<>
inline NDT_PUBLIC auto & xr_(Eigen::Vector3d & pt)
{
  return pt(0);
}

template<>
inline NDT_PUBLIC auto & yr_(Eigen::Vector3d & pt)
{
  return pt(1);
}

template<>
inline NDT_PUBLIC auto & zr_(Eigen::Vector3d & pt)
{
  return pt(2);
}

}  // namespace point_adapter
}  // namespace geometry
}  // namespace common
}  // namespace autoware

#endif  // NDT__NDT_MAP_HPP_
