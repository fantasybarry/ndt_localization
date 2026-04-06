#include "ndt_nodes/ndt_mapping_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<autoware::localization::ndt_nodes::NDTMappingNode>(
    "ndt_mapping", "localization");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
