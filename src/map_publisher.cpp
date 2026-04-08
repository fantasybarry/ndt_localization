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

#include "ndt_nodes/map_publisher.hpp"

#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <yaml-cpp/yaml.h>

#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>

#include <chrono>
#include <fstream>
#include <thread>

namespace autoware
{
namespace localization
{
namespace ndt_nodes
{

NDTMapPublisherNode::NDTMapPublisherNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("ndt_map_publisher", options)
{
  // Declare parameters.
  this->declare_parameter<std::string>("map_yaml_file", "");
  this->declare_parameter<std::string>("map_frame", "map");
  this->declare_parameter<int>("subscriber_timeout_ms", 10000);

  // Voxel grid config parameters.
  this->declare_parameter<double>("map_config.min_point.x", -500.0);
  this->declare_parameter<double>("map_config.min_point.y", -500.0);
  this->declare_parameter<double>("map_config.min_point.z", -10.0);
  this->declare_parameter<double>("map_config.max_point.x", 500.0);
  this->declare_parameter<double>("map_config.max_point.y", 500.0);
  this->declare_parameter<double>("map_config.max_point.z", 50.0);
  this->declare_parameter<double>("map_config.voxel_size.x", 1.0);
  this->declare_parameter<double>("map_config.voxel_size.y", 1.0);
  this->declare_parameter<double>("map_config.voxel_size.z", 1.0);
  this->declare_parameter<int>("map_config.capacity", 1000000);
  this->declare_parameter<bool>("viz_map", true);

  yaml_file_path_ = this->get_parameter("map_yaml_file").as_string();
  map_frame_ = this->get_parameter("map_frame").as_string();
  viz_map_ = this->get_parameter("viz_map").as_bool();

  Eigen::Vector3d min_pt(
    this->get_parameter("map_config.min_point.x").as_double(),
    this->get_parameter("map_config.min_point.y").as_double(),
    this->get_parameter("map_config.min_point.z").as_double());
  Eigen::Vector3d max_pt(
    this->get_parameter("map_config.max_point.x").as_double(),
    this->get_parameter("map_config.max_point.y").as_double(),
    this->get_parameter("map_config.max_point.z").as_double());
  Eigen::Vector3d voxel_size(
    this->get_parameter("map_config.voxel_size.x").as_double(),
    this->get_parameter("map_config.voxel_size.y").as_double(),
    this->get_parameter("map_config.voxel_size.z").as_double());
  auto capacity = static_cast<std::size_t>(
    this->get_parameter("map_config.capacity").as_int());

  map_config_ = std::make_shared<MapConfig>(min_pt, max_pt, voxel_size, capacity);

  // Publishers.
  // ndt_map uses transient_local so the localizer gets it even if it subscribes late.
  ndt_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "ndt_map", rclcpp::QoS{1}.transient_local());
  // pointcloud_map: raw XYZ cloud for PCL-based localizers (transient_local).
  raw_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "pointcloud_map", rclcpp::QoS{1}.transient_local());
  // viz_ndt_map uses reliable+volatile so RViz2 can display it without QoS config.
  viz_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "viz_ndt_map", rclcpp::QoS{1}.reliable());
  static_tf_pub_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
}

void NDTMapPublisherNode::run()
{
  // 1. Load YAML config.
  load_yaml_config();

  // 2. Publish earth → map transform.
  publish_earth_to_map_tf();

  // 3. Load PCD.
  RCLCPP_INFO(this->get_logger(), "Loading PCD: %s", pcd_file_path_.c_str());
  auto raw_cloud = load_pcd(pcd_file_path_);
  raw_cloud.header.frame_id = map_frame_;
  raw_cloud.header.stamp = this->now();

  // 4. Publish raw pointcloud map (for PCL-based localizers).
  raw_map_pub_->publish(raw_cloud);
  RCLCPP_INFO(this->get_logger(), "Published raw pointcloud map (%u points)", raw_cloud.width);

  // 5. Build NDT map.
  RCLCPP_INFO(this->get_logger(), "Building NDT map...");
  auto ndt_cloud = build_ndt_map(raw_cloud);

  // 6. Publish NDT map.
  ndt_cloud.header.frame_id = map_frame_;
  ndt_cloud.header.stamp = this->now();
  ndt_map_pub_->publish(ndt_cloud);
  RCLCPP_INFO(this->get_logger(), "Published NDT map (%u voxels)", ndt_cloud.width);

  // 6. Publish visualisation cloud (if enabled).
  if (viz_map_) {
    viz_cloud_ = downsample_for_viz(raw_cloud);
    viz_cloud_.header.frame_id = map_frame_;
    viz_cloud_.header.stamp = this->now();
    viz_map_pub_->publish(viz_cloud_);
    RCLCPP_INFO(this->get_logger(), "Published visualization map (%u points)", viz_cloud_.width);

    // Re-publish every 5 seconds so RViz can pick it up.
    viz_timer_ = this->create_wall_timer(
      std::chrono::seconds(5),
      [this]() {
        viz_cloud_.header.stamp = this->now();
        viz_map_pub_->publish(viz_cloud_);
      });
  }
}

void NDTMapPublisherNode::load_yaml_config()
{
  RCLCPP_INFO(this->get_logger(), "Loading YAML: %s", yaml_file_path_.c_str());
  YAML::Node config = YAML::LoadFile(yaml_file_path_);
  auto map_config = config["map_config"];

  pcd_file_path_ = map_config["pcd_file"].as<std::string>();
  origin_lat_ = map_config["latitude"].as<double>();
  origin_lon_ = map_config["longitude"].as<double>();
  origin_elev_ = map_config["elevation"].as<double>();
  origin_roll_ = map_config["roll"].as<double>(0.0);
  origin_pitch_ = map_config["pitch"].as<double>(0.0);
  origin_yaw_ = map_config["yaw"].as<double>(0.0);
}

sensor_msgs::msg::PointCloud2 NDTMapPublisherNode::load_pcd(const std::string & path)
{
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  if (pcl::io::loadPCDFile(path, pcl_cloud) < 0) {
    throw std::runtime_error("Failed to load PCD file: " + path);
  }

  sensor_msgs::msg::PointCloud2 ros_cloud;
  pcl::toROSMsg(pcl_cloud, ros_cloud);
  return ros_cloud;
}

sensor_msgs::msg::PointCloud2 NDTMapPublisherNode::build_ndt_map(
  const sensor_msgs::msg::PointCloud2 & cloud)
{
  // Build a DynamicNDTMap from the raw cloud.
  ndt::DynamicNDTMap dynamic_map(*map_config_);
  dynamic_map.insert(cloud);

  // Serialize each voxel into a PointCloud2 message.
  // Fields: x, y, z (centroid), icov_xx..icov_zz (inverse covariance), cell_id.
  const auto map_size = static_cast<uint32_t>(dynamic_map.size());

  sensor_msgs::msg::PointCloud2 output;
  output.height = 1;
  output.width = map_size;
  output.is_bigendian = false;
  output.is_dense = true;

  // Define fields.
  sensor_msgs::PointCloud2Modifier modifier(output);
  // We need custom fields, so build them manually.
  output.fields.clear();
  uint32_t offset = 0;

  auto add_field = [&](const std::string & name, uint8_t datatype, uint32_t count) {
    sensor_msgs::msg::PointField f;
    f.name = name;
    f.offset = offset;
    f.datatype = datatype;
    f.count = count;
    output.fields.push_back(f);
    uint32_t size = 0;
    if (datatype == sensor_msgs::msg::PointField::FLOAT64) { size = 8; }
    else if (datatype == sensor_msgs::msg::PointField::UINT32) { size = 4; }
    offset += size * count;
  };

  add_field("x", sensor_msgs::msg::PointField::FLOAT64, 1);
  add_field("y", sensor_msgs::msg::PointField::FLOAT64, 1);
  add_field("z", sensor_msgs::msg::PointField::FLOAT64, 1);
  add_field("icov_xx", sensor_msgs::msg::PointField::FLOAT64, 1);
  add_field("icov_xy", sensor_msgs::msg::PointField::FLOAT64, 1);
  add_field("icov_xz", sensor_msgs::msg::PointField::FLOAT64, 1);
  add_field("icov_yy", sensor_msgs::msg::PointField::FLOAT64, 1);
  add_field("icov_yz", sensor_msgs::msg::PointField::FLOAT64, 1);
  add_field("icov_zz", sensor_msgs::msg::PointField::FLOAT64, 1);
  add_field("cell_id", sensor_msgs::msg::PointField::UINT32, 2);

  output.point_step = offset;
  output.row_step = output.point_step * output.width;
  output.data.resize(output.row_step);

  // Fill data.
  uint32_t idx = 0;
  for (auto it = dynamic_map.cbegin(); it != dynamic_map.cend(); ++it) {
    const auto & voxel = it->second;
    if (!voxel.usable()) { continue; }

    uint8_t * ptr = output.data.data() + idx * output.point_step;

    const auto & c = voxel.centroid();
    const auto & icov = voxel.inverse_covariance();

    std::memcpy(ptr + 0, &c(0), 8);
    std::memcpy(ptr + 8, &c(1), 8);
    std::memcpy(ptr + 16, &c(2), 8);
    double icov_xx = icov(0, 0), icov_xy = icov(0, 1), icov_xz = icov(0, 2);
    double icov_yy = icov(1, 1), icov_yz = icov(1, 2), icov_zz = icov(2, 2);
    std::memcpy(ptr + 24, &icov_xx, 8);
    std::memcpy(ptr + 32, &icov_xy, 8);
    std::memcpy(ptr + 40, &icov_xz, 8);
    std::memcpy(ptr + 48, &icov_yy, 8);
    std::memcpy(ptr + 56, &icov_yz, 8);
    std::memcpy(ptr + 64, &icov_zz, 8);

    uint64_t cell_key = it->first;
    std::memcpy(ptr + 72, &cell_key, 8);

    idx++;
  }

  // Trim to actual usable voxel count.
  output.width = idx;
  output.row_step = output.point_step * output.width;
  output.data.resize(output.row_step);

  return output;
}

void NDTMapPublisherNode::publish_earth_to_map_tf()
{
  // Convert geodetic (lat/lon/elev) to ECEF using GeographicLib.
  GeographicLib::Geocentric earth(
    GeographicLib::Constants::WGS84_a(),
    GeographicLib::Constants::WGS84_f());

  double ecef_x, ecef_y, ecef_z;
  earth.Forward(origin_lat_, origin_lon_, origin_elev_, ecef_x, ecef_y, ecef_z);

  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = this->now();
  tf.header.frame_id = "earth";
  tf.child_frame_id = map_frame_;
  tf.transform.translation.x = ecef_x;
  tf.transform.translation.y = ecef_y;
  tf.transform.translation.z = ecef_z;

  // Orientation from roll/pitch/yaw.
  const double cr = std::cos(origin_roll_ * 0.5);
  const double sr = std::sin(origin_roll_ * 0.5);
  const double cp = std::cos(origin_pitch_ * 0.5);
  const double sp = std::sin(origin_pitch_ * 0.5);
  const double cy = std::cos(origin_yaw_ * 0.5);
  const double sy = std::sin(origin_yaw_ * 0.5);

  tf.transform.rotation.w = cr * cp * cy + sr * sp * sy;
  tf.transform.rotation.x = sr * cp * cy - cr * sp * sy;
  tf.transform.rotation.y = cr * sp * cy + sr * cp * sy;
  tf.transform.rotation.z = cr * cp * sy - sr * sp * cy;

  static_tf_pub_->sendTransform(tf);
  RCLCPP_INFO(
    this->get_logger(),
    "Published earth->%s TF (ECEF: %.2f, %.2f, %.2f)",
    map_frame_.c_str(), ecef_x, ecef_y, ecef_z);
}

sensor_msgs::msg::PointCloud2 NDTMapPublisherNode::downsample_for_viz(
  const sensor_msgs::msg::PointCloud2 & cloud)
{
  // Simple pass-through for now. A voxel grid filter can be added later.
  // Convert to float32 xyz-only for RViz.
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  pcl::fromROSMsg(cloud, pcl_cloud);

  sensor_msgs::msg::PointCloud2 output;
  pcl::toROSMsg(pcl_cloud, output);
  return output;
}

}  // namespace ndt_nodes
}  // namespace localization
}  // namespace autoware
