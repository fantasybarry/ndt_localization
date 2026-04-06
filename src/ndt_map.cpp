#include "ndt/ndt_map.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <string>

namespace autoware
{
namespace localization
{
namespace ndt
{

/// Validate that a PointCloud2 message has the required fields for a static NDT map.
/// Returns the number of points if valid, 0 otherwise.
uint32_t validate_pcl_map(const sensor_msgs::msg::PointCloud2 & msg)
{
  if (msg.width == 0U) {
    return 0U;
  }

  // Check that required fields exist.
  const std::vector<std::string> required_fields = {
    "x", "y", "z",
    "icov_xx", "icov_xy", "icov_xz",
    "icov_yy", "icov_yz", "icov_zz",
    "cell_id"
  };

  for (const auto & name : required_fields) {
    bool found = false;
    for (const auto & field : msg.fields) {
      if (field.name == name) {
        found = true;
        break;
      }
    }
    if (!found) {
      return 0U;
    }
  }

  return msg.width;
}

}  // namespace ndt
}  // namespace localization
}  // namespace autoware
