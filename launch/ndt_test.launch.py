import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('ndt_localization')

    map_publisher_param = os.path.join(pkg_dir, 'param', 'map_publisher.param.yaml')
    localizer_param = os.path.join(pkg_dir, 'param', 'ndt_localizer.param.yaml')

    map_publisher_node = Node(
        package='ndt_localization',
        executable='ndt_map_publisher_exe',
        name='ndt_map_publisher',
        namespace='localization',
        parameters=[map_publisher_param],
        output='screen',
    )

    localizer_node = Node(
        package='ndt_localization',
        executable='p2d_ndt_localizer_exe',
        name='p2d_ndt_localizer',
        namespace='localization',
        parameters=[localizer_param],
        remappings=[
            ('points_raw', '/velodyne_points'),
            ('pointcloud_map', '/localization/pointcloud_map'),
            ('imu_raw', '/imu/data'),
        ],
        output='screen',
    )

    # Static TF: base_link -> velodyne (adjust if lidar is offset from base)
    base_to_velodyne_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'velodyne'],
        output='screen',
    )

    return LaunchDescription([
        map_publisher_node,
        localizer_node,
        base_to_velodyne_tf,
    ])
