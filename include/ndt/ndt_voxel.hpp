#ifndef NDT__NDT_VOXEL_HPP_
#define NDT__NDT_VOXEL_HPP_

#include <ndt/visibility_control.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace autoware
{
namespace localization
{
namespace ndt
{

// ---------------------------------------------------------------------------
// DynamicNDTVoxel — online voxel updated with Welford's algorithm
// ---------------------------------------------------------------------------

/// A voxel that accumulates observations and computes centroid + covariance
/// incrementally (Welford's online algorithm).
class NDT_PUBLIC DynamicNDTVoxel
{
public:
  DynamicNDTVoxel() = default;

  /// Add a single 3-D point observation.
  void add_observation(const Eigen::Vector3d & pt)
  {
    ++m_count;
    Eigen::Vector3d delta = pt - m_mean;
    m_mean += delta / static_cast<double>(m_count);
    Eigen::Vector3d delta2 = pt - m_mean;
    m_m2 += delta * delta2.transpose();
  }

  /// Attempt to stabilise (finalise) the covariance.
  /// Returns true if the voxel has enough points to be usable.
  bool try_stabilize()
  {
    if (m_count >= 3) {
      m_covariance = m_m2 / static_cast<double>(m_count - 1);
      // Regularise to prevent singular covariance.
      m_covariance.diagonal().array() += 1e-3;
      m_inverse_covariance = m_covariance.inverse();
      m_stable = true;
    }
    return m_stable;
  }

  /// Whether the voxel has enough data to produce a valid covariance.
  bool usable() const { return m_stable && m_count >= 3; }

  const Eigen::Vector3d & centroid() const { return m_mean; }
  const Eigen::Matrix3d & covariance() const { return m_covariance; }
  const Eigen::Matrix3d & inverse_covariance() const { return m_inverse_covariance; }
  int64_t count() const { return m_count; }

private:
  int64_t m_count = 0;
  Eigen::Vector3d m_mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d m_m2 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d m_covariance = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d m_inverse_covariance = Eigen::Matrix3d::Zero();
  bool m_stable = false;
};

// ---------------------------------------------------------------------------
// StaticNDTVoxel — pre-computed voxel loaded from a serialised map
// ---------------------------------------------------------------------------

/// A voxel with a pre-computed centroid and inverse covariance.
/// No online update is performed.
class NDT_PUBLIC StaticNDTVoxel
{
public:
  StaticNDTVoxel() = default;

  StaticNDTVoxel(
    const Eigen::Vector3d & centroid,
    const Eigen::Matrix3d & inverse_covariance)
  : m_centroid(centroid),
    m_inverse_covariance(inverse_covariance),
    m_usable(true) {}

  bool usable() const { return m_usable; }

  const Eigen::Vector3d & centroid() const { return m_centroid; }
  const Eigen::Matrix3d & inverse_covariance() const { return m_inverse_covariance; }

private:
  Eigen::Vector3d m_centroid = Eigen::Vector3d::Zero();
  Eigen::Matrix3d m_inverse_covariance = Eigen::Matrix3d::Zero();
  bool m_usable = false;
};

}  // namespace ndt
}  // namespace localization
}  // namespace autoware

#endif  // NDT__NDT_VOXEL_HPP_
