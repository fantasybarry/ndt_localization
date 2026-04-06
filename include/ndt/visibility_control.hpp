#ifndef NDT__VISIBILITY_CONTROL_HPP_
#define NDT__VISIBILITY_CONTROL_HPP_

#if defined(__GNUC__) || defined(__clang__)
  #define NDT_EXPORT __attribute__((visibility("default")))
  #define NDT_IMPORT
#elif defined(_MSC_VER)
  #define NDT_EXPORT __declspec(dllexport)
  #define NDT_IMPORT __declspec(dllimport)
#else
  #define NDT_EXPORT
  #define NDT_IMPORT
#endif

#ifdef NDT_BUILDING_DLL
  #define NDT_PUBLIC NDT_EXPORT
#else
  #define NDT_PUBLIC NDT_IMPORT
#endif

#define NDT_LOCAL

#endif  // NDT__VISIBILITY_CONTROL_HPP_
