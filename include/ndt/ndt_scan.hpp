#ifndef NDT__NDT_SCAN_HPP_
#define NDT__NDT_SCAN_HPP_

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>

namespace ndt
{

/// Wraps a vector of 3-D points and exposes iterators so that the optimisation
/// routine can walk over scan points without knowing the underlying storage.
class P2DNDTScan
{
public:
  using PointVec = std::vector<Eigen::Vector3d>;
  using const_iterator = PointVec::const_iterator;

  P2DNDTScan() = default;

  /// Construct from a PCL point cloud, applying an optional voxel down-sample.
  explicit P2DNDTScan(
    const pcl::PointCloud<pcl::PointXYZ> & cloud,
    double downsample_leaf = 0.0);

  /// Construct directly from an Eigen vector.
  explicit P2DNDTScan(PointVec && points);

  const_iterator begin() const { return points_.begin(); }
  const_iterator end() const { return points_.end(); }
  std::size_t size() const { return points_.size(); }
  bool empty() const { return points_.empty(); }

  const Eigen::Vector3d & operator[](std::size_t i) const { return points_[i]; }

private:
  PointVec points_;
};

}  // namespace ndt

#endif  // NDT__NDT_SCAN_HPP_
