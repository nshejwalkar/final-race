import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pp_share = get_package_share_directory('pure_pursuit')
    pf_share = get_package_share_directory('particle_filter')

    waypoints_path_arg = DeclareLaunchArgument(
        'waypoints_path',
        default_value='',
        description='Absolute path to waypoint CSV. Leave empty to use installed default.',
    )

    pure_pursuit_node = Node(
        package='pure_pursuit',
        executable='pure_pursuit_node.py',
        name='pure_pursuit_node',
        parameters=[
            os.path.join(pp_share, 'config', 'pure_pursuit.yaml'),
            {
                'waypoints_path': LaunchConfiguration('waypoints_path'),
                'pose_topic': '/pf/pose/odom',
            },
        ],
        output='screen',
    )

    particle_filter_node = Node(
        package='particle_filter',
        executable='particle_filter_node.py',
        name='particle_filter',
        parameters=[os.path.join(pf_share, 'config', 'localize.yaml')],
        output='screen',
    )

    return LaunchDescription([
        waypoints_path_arg,
        particle_filter_node,
        pure_pursuit_node,
    ])
