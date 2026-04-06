#ifndef NDT__NDT_VOXEL_VIEW_HPP_
#define NDT__NDT_VOXEL_VIEW_HPP_

#include <ndt/visibility_control.hpp>

#include <Eigen/Core>

namespace autoware
{
namespace localization
{
namespace ndt
{

/// Lightweight, non-owning view over a voxel.
/// Provides a uniform read-only interface regardless of whether the
/// underlying voxel is Dynamic or Static.
template <typename VoxelT>
class VoxelView
{
public:
  explicit VoxelView(const VoxelT & voxel)
  : m_voxel(&voxel) {}

  bool usable() const { return m_voxel->usable(); }
  const Eigen::Vector3d & centroid() const { return m_voxel->centroid(); }
  const Eigen::Matrix3d & inverse_covariance() const
  {
    return m_voxel->inverse_covariance();
  }

private:
  const VoxelT * m_voxel;
};

}  // namespace ndt
}  // namespace localization
}  // namespace autoware

#endif  // NDT__NDT_VOXEL_VIEW_HPP_
