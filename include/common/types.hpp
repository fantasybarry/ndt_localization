#ifndef COMMON__TYPES_HPP_
#define COMMON__TYPES_HPP_

#include <cstdint>

namespace autoware
{
namespace common
{
namespace types
{

using bool8_t = bool;
using float32_t = float;
using float64_t = double;

/// The NDT scalar type used throughout the localization stack.
using Real = float64_t;

}  // namespace types
}  // namespace common
}  // namespace autoware

#endif  // COMMON__TYPES_HPP_
