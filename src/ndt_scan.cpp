#include "ndt/ndt_scan.hpp"

#include <pcl/filters/voxel_grid.h>

namespace ndt
{

P2DNDTScan::P2DNDTScan(
  const pcl::PointCloud<pcl::PointXYZ> & cloud,
  double downsample_leaf)
{
  pcl::PointCloud<pcl::PointXYZ> filtered;

  if (downsample_leaf > 0.0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud.makeShared());
    vg.setLeafSize(
      static_cast<float>(downsample_leaf),
      static_cast<float>(downsample_leaf),
      static_cast<float>(downsample_leaf));
    vg.filter(filtered);
  } else {
    filtered = cloud;
  }

  points_.reserve(filtered.size());
  for (const auto & p : filtered.points) {
    points_.emplace_back(p.x, p.y, p.z);
  }
}

P2DNDTScan::P2DNDTScan(PointVec && points)
: points_(std::move(points))
{
}

}  // namespace ndt
