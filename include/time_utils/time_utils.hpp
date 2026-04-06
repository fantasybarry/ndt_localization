#ifndef TIME_UTILS__TIME_UTILS_HPP_
#define TIME_UTILS__TIME_UTILS_HPP_

#include <builtin_interfaces/msg/time.hpp>
#include <chrono>

namespace time_utils
{

/// Convert a ROS builtin_interfaces Time message to a system_clock time_point.
inline std::chrono::system_clock::time_point from_message(
  const builtin_interfaces::msg::Time & stamp)
{
  auto duration = std::chrono::seconds(stamp.sec) +
    std::chrono::nanoseconds(stamp.nanosec);
  return std::chrono::system_clock::time_point(
    std::chrono::duration_cast<std::chrono::system_clock::duration>(duration));
}

}  // namespace time_utils

#endif  // TIME_UTILS__TIME_UTILS_HPP_
