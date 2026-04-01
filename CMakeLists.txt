cmake_minimum_required(VERSION 3.14)
project(ndt_localization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

add_library(ndt
  src/ndt_map.cpp
  src/ndt_scan.cpp
  src/ndt_optimization.cpp
)

target_include_directories(ndt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(ndt PUBLIC
  Eigen3::Eigen
  ${PCL_LIBRARIES}
)
