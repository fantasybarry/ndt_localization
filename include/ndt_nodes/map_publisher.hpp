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

#ifndef NDT_NODES__MAP_PUBLISHER_HPP_
#define NDT_NODES__MAP_PUBLISHER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <ndt/ndt_map.hpp>

#include <string>
#include <memory>

namespace autoware
{
namespace localization
{
namespace ndt_nodes
{

using autoware::perception::filters::voxel_grid::Config;

/// ROS 2 node that loads a PCD map, converts it to an NDT representation,
/// and publishes the result.
///
/// Workflow (run()):
///   1. Wait for subscribers on ndt_map / viz_ndt_map
///   2. Parse YAML for PCD filename + geodetic origin
///   3. Convert geodetic → ECEF, publish earth→map static TF
///   4. Load PCD → PointCloud2
///   5. Build DynamicNDTMap
///   6. Serialize NDT map → PointCloud2 (x,y,z + icov + cell_id)
///   7. Publish ndt_map
///   8. Downsample original cloud for visualisation
///   9. Publish viz_ndt_map
class NDTMapPublisherNode : public rclcpp::Node
{
public:
  using SerializedMap = ndt::StaticNDTMap;
  using MapConfig = Config;

  explicit NDTMapPublisherNode(const rclcpp::NodeOptions & options);

  /// Execute the full map loading and publishing pipeline.
  void run();

private:
  /// Load the YAML config file and extract PCD path + geodetic origin.
  void load_yaml_config();

  /// Load a PCD file into a PointCloud2 message.
  sensor_msgs::msg::PointCloud2 load_pcd(const std::string & path);

  /// Build a DynamicNDTMap from a PointCloud2, then serialize it back out
  /// as a PointCloud2 with NDT voxel fields.
  sensor_msgs::msg::PointCloud2 build_ndt_map(
    const sensor_msgs::msg::PointCloud2 & cloud);

  /// Publish the earth→map static transform from geodetic coordinates.
  void publish_earth_to_map_tf();

  /// Downsample a point cloud for visualisation.
  sensor_msgs::msg::PointCloud2 downsample_for_viz(
    const sensor_msgs::msg::PointCloud2 & cloud);

  // Publishers.
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ndt_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr viz_map_pub_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_pub_;

  // Parameters.
  std::string yaml_file_path_;
  std::string pcd_file_path_;
  std::string map_frame_;

  // Geodetic origin from YAML.
  double origin_lat_ = 0.0;
  double origin_lon_ = 0.0;
  double origin_elev_ = 0.0;
  double origin_roll_ = 0.0;
  double origin_pitch_ = 0.0;
  double origin_yaw_ = 0.0;

  // Voxel grid config for the NDT map.
  std::shared_ptr<MapConfig> map_config_;
};

}  // namespace ndt_nodes
}  // namespace localization
}  // namespace autoware

#endif  // NDT_NODES__MAP_PUBLISHER_HPP_
