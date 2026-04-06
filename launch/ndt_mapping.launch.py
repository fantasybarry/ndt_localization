import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('ndt_localization')
    mapping_param = os.path.join(pkg_dir, 'param', 'ndt_mapping.param.yaml')

    mapping_node = Node(
        package='ndt_localization',
        executable='ndt_mapping_exe',
        name='ndt_mapping',
        namespace='localization',
        parameters=[mapping_param],
        remappings=[
            ('points_raw', '/sensing/lidar/points_raw'),
            ('imu_raw', '/sensing/imu/imu_raw'),
            ('odom', '/vehicle/odom'),
        ],
        output='screen',
    )

    return LaunchDescription([
        mapping_node,
    ])
