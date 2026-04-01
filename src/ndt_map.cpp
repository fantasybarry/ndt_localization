#include "ndt/ndt_map.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <algorithm>
#include <string>

#include <pcl/io/pcd_io.h>

#include <cmath>

namespace ndt
{

// ===========================================================================
// DynamicNDTMap
// ===========================================================================

DynamicNDTMap::DynamicNDTMap(const Eigen::Vector3d & voxel_size)
: voxel_size_(voxel_size)
{
}

Eigen::Vector3i DynamicNDTMap::to_index(const Eigen::Vector3d & point) const
{
  return Eigen::Vector3i(
    static_cast<int>(std::floor(point.x() / voxel_size_.x())),
    static_cast<int>(std::floor(point.y() / voxel_size_.y())),
    static_cast<int>(std::floor(point.z() / voxel_size_.z())));
}

void DynamicNDTMap::insert(const pcl::PointCloud<pcl::PointXYZ> & cloud)
{
  for (const auto & pcl_pt : cloud.points) {
    Eigen::Vector3d pt(pcl_pt.x, pcl_pt.y, pcl_pt.z);
    auto idx = to_index(pt);
    auto & acc = accumulators_[idx];

    // Welford's online algorithm for mean and covariance.
    acc.n++;
    Eigen::Vector3d delta = pt - acc.mean;
    acc.mean += delta / static_cast<double>(acc.n);
    Eigen::Vector3d delta2 = pt - acc.mean;
    acc.m2 += delta * delta2.transpose();
  }

  // Finalise every touched voxel.
  for (auto & [idx, acc] : accumulators_) {
    finalize_voxel(idx);
  }
}

void DynamicNDTMap::finalize_voxel(const Eigen::Vector3i & idx)
{
  const auto & acc = accumulators_.at(idx);
  Voxel & v = grid_[idx];
  v.centroid = acc.mean;
  v.point_count = acc.n;

  if (acc.n >= 3) {
    v.covariance = acc.m2 / static_cast<double>(acc.n - 1);
    // Regularise to avoid singular covariance.
    v.covariance.diagonal().array() += 1e-3;
  }
}

void DynamicNDTMap::clear()
{
  accumulators_.clear();
  grid_.clear();
}

const Voxel * DynamicNDTMap::lookup(const Eigen::Vector3d & point) const
{
  auto it = grid_.find(to_index(point));
  return (it != grid_.end() && it->second.usable()) ? &it->second : nullptr;
}

std::vector<const Voxel *> DynamicNDTMap::nearby_voxels(
  const Eigen::Vector3d & point, double radius) const
{
  std::vector<const Voxel *> result;
  int rx = static_cast<int>(std::ceil(radius / voxel_size_.x()));
  int ry = static_cast<int>(std::ceil(radius / voxel_size_.y()));
  int rz = static_cast<int>(std::ceil(radius / voxel_size_.z()));
  Eigen::Vector3i center = to_index(point);

  for (int dx = -rx; dx <= rx; ++dx) {
    for (int dy = -ry; dy <= ry; ++dy) {
      for (int dz = -rz; dz <= rz; ++dz) {
        Eigen::Vector3i idx = center + Eigen::Vector3i(dx, dy, dz);
        auto it = grid_.find(idx);
        if (it != grid_.end() && it->second.usable()) {
          result.push_back(&it->second);
        }
      }
    }
  }
  return result;
}

// ===========================================================================
// StaticNDTMap
// ===========================================================================

Eigen::Vector3i StaticNDTMap::to_index(const Eigen::Vector3d & point) const
{
  return Eigen::Vector3i(
    static_cast<int>(std::floor(point.x() / voxel_size_.x())),
    static_cast<int>(std::floor(point.y() / voxel_size_.y())),
    static_cast<int>(std::floor(point.z() / voxel_size_.z())));
}

bool StaticNDTMap::load_from_pcd(
  const std::string & pcd_path, const Eigen::Vector3d & voxel_size)
{
  voxel_size_ = voxel_size;
  grid_.clear();

  // Load raw point cloud and build NDT via simple batch statistics per voxel.
  pcl::PointCloud<pcl::PointXYZ> cloud;
  if (pcl::io::loadPCDFile(pcd_path, cloud) < 0) {
    return false;
  }

  // Accumulate points into voxels.
  std::unordered_map<Eigen::Vector3i, std::vector<Eigen::Vector3d>, VoxelKeyHash> buckets;
  for (const auto & p : cloud.points) {
    Eigen::Vector3d pt(p.x, p.y, p.z);
    buckets[to_index(pt)].push_back(pt);
  }

  for (auto & [idx, pts] : buckets) {
    if (static_cast<int64_t>(pts.size()) < 3) {
      continue;
    }
    Voxel v;
    v.point_count = static_cast<int64_t>(pts.size());

    // Compute centroid.
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto & p : pts) { sum += p; }
    v.centroid = sum / static_cast<double>(pts.size());

    // Compute covariance.
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto & p : pts) {
      Eigen::Vector3d d = p - v.centroid;
      cov += d * d.transpose();
    }
    v.covariance = cov / static_cast<double>(pts.size() - 1);
    v.covariance.diagonal().array() += 1e-3;

    grid_[idx] = v;
  }

  return true;
}

const Voxel * StaticNDTMap::lookup(const Eigen::Vector3d & point) const
{
  auto it = grid_.find(to_index(point));
  return (it != grid_.end() && it->second.usable()) ? &it->second : nullptr;
}

std::vector<const Voxel *> StaticNDTMap::nearby_voxels(
  const Eigen::Vector3d & point, double radius) const
{
  std::vector<const Voxel *> result;
  int rx = static_cast<int>(std::ceil(radius / voxel_size_.x()));
  int ry = static_cast<int>(std::ceil(radius / voxel_size_.y()));
  int rz = static_cast<int>(std::ceil(radius / voxel_size_.z()));
  Eigen::Vector3i center = to_index(point);

  for (int dx = -rx; dx <= rx; ++dx) {
    for (int dy = -ry; dy <= ry; ++dy) {
      for (int dz = -rz; dz <= rz; ++dz) {
        Eigen::Vector3i idx = center + Eigen::Vector3i(dx, dy, dz);
        auto it = grid_.find(idx);
        if (it != grid_.end() && it->second.usable()) {
          result.push_back(&it->second);
        }
      }
    }
  }
  return result;
}

}  // namespace ndt
