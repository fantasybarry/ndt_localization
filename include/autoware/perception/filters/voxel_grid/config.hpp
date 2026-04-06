#ifndef AUTOWARE__PERCEPTION__FILTERS__VOXEL_GRID__CONFIG_HPP_
#define AUTOWARE__PERCEPTION__FILTERS__VOXEL_GRID__CONFIG_HPP_

#include <Eigen/Core>

#include <cmath>
#include <cstdint>

namespace autoware
{
namespace perception
{
namespace filters
{
namespace voxel_grid
{

/// Minimal voxel grid configuration.
/// Maps a 3-D point to a flat uint64_t index suitable for use as a hash key.
class Config
{
public:
  Config(
    const Eigen::Vector3d & min_point,
    const Eigen::Vector3d & max_point,
    const Eigen::Vector3d & voxel_size,
    std::size_t capacity)
  : min_point_(min_point),
    max_point_(max_point),
    voxel_size_(voxel_size),
    capacity_(capacity)
  {
    dims_x_ = static_cast<uint64_t>(
      std::ceil((max_point_.x() - min_point_.x()) / voxel_size_.x()));
    dims_y_ = static_cast<uint64_t>(
      std::ceil((max_point_.y() - min_point_.y()) / voxel_size_.y()));
  }

  /// Compute a flat voxel index for a given 3-D point.
  uint64_t index(const Eigen::Vector3d & pt) const
  {
    const uint64_t ix = static_cast<uint64_t>(
      std::floor((pt.x() - min_point_.x()) / voxel_size_.x()));
    const uint64_t iy = static_cast<uint64_t>(
      std::floor((pt.y() - min_point_.y()) / voxel_size_.y()));
    const uint64_t iz = static_cast<uint64_t>(
      std::floor((pt.z() - min_point_.z()) / voxel_size_.z()));
    return ix + iy * dims_x_ + iz * dims_x_ * dims_y_;
  }

  std::size_t get_capacity() const { return capacity_; }
  const Eigen::Vector3d & get_voxel_size() const { return voxel_size_; }

private:
  Eigen::Vector3d min_point_;
  Eigen::Vector3d max_point_;
  Eigen::Vector3d voxel_size_;
  std::size_t capacity_;
  uint64_t dims_x_ = 0;
  uint64_t dims_y_ = 0;
};

}  // namespace voxel_grid
}  // namespace filters
}  // namespace perception
}  // namespace autoware

#endif  // AUTOWARE__PERCEPTION__FILTERS__VOXEL_GRID__CONFIG_HPP_
