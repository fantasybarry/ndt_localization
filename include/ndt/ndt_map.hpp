#ifndef NDT__NDT_MAP_HPP_
#define NDT__NDT_MAP_HPP_

#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <unordered_map>
#include <vector>

namespace ndt
{

/// A single NDT voxel storing a centroid and covariance computed from contained points.
struct Voxel
{
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  int64_t point_count = 0;

  /// Whether the voxel has enough points to be usable (minimum 3 for a 3D covariance).
  bool usable() const { return point_count >= 3; }
};

/// Hash function for discretised 3-D grid indices.
struct VoxelKeyHash
{
  std::size_t operator()(const Eigen::Vector3i & key) const
  {
    // Combine hashes of individual components.
    std::size_t seed = std::hash<int>()(key.x());
    seed ^= std::hash<int>()(key.y()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(key.z()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

inline bool operator==(const Eigen::Vector3i & a, const Eigen::Vector3i & b)
{
  return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
}

// ---------------------------------------------------------------------------
// DynamicNDTMap
// ---------------------------------------------------------------------------

/// Builds an NDT representation on-the-fly from incoming point clouds using
/// Welford's online algorithm for incremental mean and covariance computation.
class DynamicNDTMap
{
public:
  explicit DynamicNDTMap(const Eigen::Vector3d & voxel_size);

  /// Insert a point cloud into the map, updating voxel statistics incrementally.
  void insert(const pcl::PointCloud<pcl::PointXYZ> & cloud);

  /// Clear all voxel data.
  void clear();

  /// Look up the voxel that contains the given 3-D point.
  /// Returns nullptr when no voxel exists at that location.
  const Voxel * lookup(const Eigen::Vector3d & point) const;

  /// Collect all usable voxels whose centroids fall within a given radius of a query point.
  std::vector<const Voxel *> nearby_voxels(
    const Eigen::Vector3d & point, double radius) const;

  std::size_t size() const { return grid_.size(); }

private:
  Eigen::Vector3i to_index(const Eigen::Vector3d & point) const;

  Eigen::Vector3d voxel_size_;

  // Intermediate accumulators for Welford's algorithm.
  struct Accumulator
  {
    int64_t n = 0;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d m2 = Eigen::Matrix3d::Zero();  // sum of outer products of diffs
  };

  std::unordered_map<Eigen::Vector3i, Accumulator, VoxelKeyHash> accumulators_;
  std::unordered_map<Eigen::Vector3i, Voxel, VoxelKeyHash> grid_;

  /// Recompute the Voxel from its Accumulator.
  void finalize_voxel(const Eigen::Vector3i & idx);
};

// ---------------------------------------------------------------------------
// StaticNDTMap
// ---------------------------------------------------------------------------

/// Loads a pre-computed NDT map (voxel centroids + covariances) from a serialised
/// point cloud.  No online update is performed.
class StaticNDTMap
{
public:
  StaticNDTMap() = default;

  /// Load from a PCD file where each "point" encodes centroid + upper-triangle covariance.
  bool load_from_pcd(const std::string & pcd_path, const Eigen::Vector3d & voxel_size);

  const Voxel * lookup(const Eigen::Vector3d & point) const;

  std::vector<const Voxel *> nearby_voxels(
    const Eigen::Vector3d & point, double radius) const;

  std::size_t size() const { return grid_.size(); }

private:
  Eigen::Vector3i to_index(const Eigen::Vector3d & point) const;

  Eigen::Vector3d voxel_size_ = Eigen::Vector3d::Ones();
  std::unordered_map<Eigen::Vector3i, Voxel, VoxelKeyHash> grid_;
};

}  // namespace ndt

#endif  // NDT__NDT_MAP_HPP_
