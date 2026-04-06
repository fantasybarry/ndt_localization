#include "ndt_nodes/ndt_localizer_nodes.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  using LocalizerNode = autoware::localization::ndt_nodes::P2DNDTLocalizerNode<>;

  auto node = std::make_shared<LocalizerNode>(
    "p2d_ndt_localizer", "localization");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
