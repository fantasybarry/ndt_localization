#ifndef NDT__NDT_SCAN_HPP_
#define NDT__NDT_SCAN_HPP_

#include <helper_functions/crtp.hpp>
#include <ndt/visibility_control.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <time_utils/time_utils.hpp>
#include <Eigen/Core>
#include <vector>
#include "common/types.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using autoware::common::types::bool8_t;
using autoware::common::types::float32_t;

namespace autoware
{
namespace localization
{
namespace ndt
{

// ---------------------------------------------------------------------------
// NDTScanBase — CRTP base class
// ---------------------------------------------------------------------------

template<typename Derived, typename NDTUnit, typename IteratorT>
class NDTScanBase : public common::helper_functions::crtp<Derived>
{
public:
  using Point = NDTUnit;
  using TimePoint = std::chrono::system_clock::time_point;

  IteratorT begin() const { return this->impl().begin_(); }
  IteratorT end() const { return this->impl().end_(); }

  void clear() { return this->impl().clear_(); }
  bool8_t empty() { return this->impl().empty_(); }

  void insert(const sensor_msgs::msg::PointCloud2 & msg)
  {
    this->impl().insert_(msg);
  }

  /// Insert from a PCL point cloud (non-ROS path).
  void insert(const pcl::PointCloud<pcl::PointXYZ> & cloud)
  {
    this->impl().insert_pcl_(cloud);
  }

  std::size_t size() const { return this->impl().size_(); }
  TimePoint stamp() { return this->impl().stamp_(); }
};

// ---------------------------------------------------------------------------
// P2DNDTScan — concrete scan for Point-to-Distribution NDT
// ---------------------------------------------------------------------------

class NDT_PUBLIC P2DNDTScan : public NDTScanBase<P2DNDTScan,
    Eigen::Vector3d, std::vector<Eigen::Vector3d>::const_iterator>
{
public:
  using Container = std::vector<Eigen::Vector3d>;
  using iterator = Container::const_iterator;

  static_assert(
    std::is_same<decltype(std::declval<NDTScanBase>().begin()), iterator>::value,
    "P2DNDTScan: The iterator type parameter should match the "
    "iterator of the container.");

  // Scans should be moved rather than being copied.
  P2DNDTScan(const P2DNDTScan &) = delete;
  P2DNDTScan & operator=(const P2DNDTScan &) = delete;
  P2DNDTScan(P2DNDTScan &&) = default;
  P2DNDTScan & operator=(P2DNDTScan &&) = default;

  /// Construct with a pre-allocated capacity.
  explicit P2DNDTScan(std::size_t capacity)
  {
    m_points.reserve(capacity);
  }

  /// Construct from a PointCloud2 message with a capacity limit.
  P2DNDTScan(
    const sensor_msgs::msg::PointCloud2 & msg,
    std::size_t capacity)
  {
    m_points.reserve(capacity);
    insert_(msg);
  }

  /// Construct from a PCL point cloud (kept from our implementation).
  P2DNDTScan(
    const pcl::PointCloud<pcl::PointXYZ> & cloud,
    std::size_t capacity)
  {
    m_points.reserve(capacity);
    insert_pcl_(cloud);
  }

  /// Insert from a ROS PointCloud2 message.
  void insert_(const sensor_msgs::msg::PointCloud2 & msg)
  {
    if (!m_points.empty()) {
      m_points.clear();
    }

    m_stamp = ::time_utils::from_message(msg.header.stamp);

    constexpr auto container_full_error = "received a lidar scan with more points than the "
      "ndt scan representation can contain. Please re-configure the scan "
      "representation accordingly.";

    if (msg.width > m_points.capacity()) {
      throw std::length_error(container_full_error);
    }

    sensor_msgs::PointCloud2ConstIterator<float32_t> x_it(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float32_t> y_it(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float32_t> z_it(msg, "z");

    while (x_it != x_it.end() &&
      y_it != y_it.end() &&
      z_it != z_it.end())
    {
      if (m_points.size() == m_points.capacity()) {
        throw std::length_error(container_full_error);
      }
      m_points.emplace_back(*x_it, *y_it, *z_it);
      ++x_it;
      ++y_it;
      ++z_it;
    }
  }

  /// Insert from a PCL point cloud (kept from our implementation).
  void insert_pcl_(const pcl::PointCloud<pcl::PointXYZ> & cloud)
  {
    if (!m_points.empty()) {
      m_points.clear();
    }

    constexpr auto container_full_error = "received a lidar scan with more points than the "
      "ndt scan representation can contain. Please re-configure the scan "
      "representation accordingly.";

    if (cloud.size() > m_points.capacity()) {
      throw std::length_error(container_full_error);
    }

    for (const auto & p : cloud.points) {
      if (m_points.size() == m_points.capacity()) {
        throw std::length_error(container_full_error);
      }
      m_points.emplace_back(p.x, p.y, p.z);
    }
  }

  iterator begin_() const { return m_points.cbegin(); }
  iterator end_() const { return m_points.cend(); }
  bool8_t empty_() { return m_points.empty(); }
  void clear_() { m_points.clear(); }
  std::size_t size_() const { return m_points.size(); }
  TimePoint stamp_() { return m_stamp; }

  const Eigen::Vector3d & operator[](std::size_t i) const { return m_points[i]; }

private:
  Container m_points;
  NDTScanBase::TimePoint m_stamp{};
};

}  // namespace ndt
}  // namespace localization
}  // namespace autoware

#endif  // NDT__NDT_SCAN_HPP_
