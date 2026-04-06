cmake_minimum_required(VERSION 3.14)
project(ndt_localization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(yaml-cpp REQUIRED)

# Try to find GeographicLib via cmake config or pkg-config.
find_package(GeographicLib QUIET)
if(NOT GeographicLib_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GeographicLib REQUIRED geographiclib)
endif()

# ---------------------------------------------------------------------------
# NDT core library
# ---------------------------------------------------------------------------
add_library(ndt
  src/ndt_map.cpp
  src/ndt_optimization.cpp
  src/ndt_scan.cpp
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

ament_target_dependencies(ndt PUBLIC
  rclcpp
  sensor_msgs
  geometry_msgs
  builtin_interfaces
)

# ---------------------------------------------------------------------------
# NDT Map Publisher Node
# ---------------------------------------------------------------------------
add_library(ndt_map_publisher_lib
  src/map_publisher.cpp
)

target_include_directories(ndt_map_publisher_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(ndt_map_publisher_lib PUBLIC
  ndt
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  yaml-cpp
  ${GeographicLib_LIBRARIES}
)

ament_target_dependencies(ndt_map_publisher_lib PUBLIC
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  pcl_conversions
)

add_executable(ndt_map_publisher_exe
  src/map_publisher_main.cpp
)

target_link_libraries(ndt_map_publisher_exe
  ndt_map_publisher_lib
)

# ---------------------------------------------------------------------------
# P2D NDT Localizer Node
# ---------------------------------------------------------------------------
add_executable(p2d_ndt_localizer_exe
  src/ndt_localizer_main.cpp
)

target_include_directories(p2d_ndt_localizer_exe PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(p2d_ndt_localizer_exe
  ndt
  Eigen3::Eigen
)

ament_target_dependencies(p2d_ndt_localizer_exe
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS
  ndt_map_publisher_exe
  p2d_ndt_localizer_exe
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY param/
  DESTINATION share/${PROJECT_NAME}/param
)

ament_package()
