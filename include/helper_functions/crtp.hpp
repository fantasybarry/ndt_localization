#ifndef HELPER_FUNCTIONS__CRTP_HPP_
#define HELPER_FUNCTIONS__CRTP_HPP_

namespace autoware
{
namespace common
{
namespace helper_functions
{

/// Curiously Recurring Template Pattern helper.
/// Provides a safe impl() cast from base to derived.
template <typename Derived>
class crtp
{
protected:
  Derived & impl() { return static_cast<Derived &>(*this); }
  const Derived & impl() const { return static_cast<const Derived &>(*this); }
};

}  // namespace helper_functions
}  // namespace common
}  // namespace autoware

#endif  // HELPER_FUNCTIONS__CRTP_HPP_
