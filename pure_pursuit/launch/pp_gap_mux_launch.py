import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory('pure_pursuit')
    config = os.path.join(share, 'config', 'pure_pursuit.yaml')

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
            config,
            {'waypoints_path': LaunchConfiguration('waypoints_path')},
        ],
        output='screen',
    )

    gap_follow_node = Node(
        package='pure_pursuit',
        executable='gap_follow.py',
        name='gap_follow',
        output='screen',
    )

    drive_mux_node = Node(
        package='pure_pursuit',
        executable='drive_mux.py',
        name='drive_mux',
        output='screen',
    )

    return LaunchDescription([
        waypoints_path_arg,
        pure_pursuit_node,
        gap_follow_node,
        drive_mux_node,
    ])
