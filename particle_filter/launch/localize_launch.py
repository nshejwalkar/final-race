import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('particle_filter'),
        'config',
        'localize.yaml',
    )

    particle_filter_node = Node(
        package='particle_filter',
        executable='particle_filter_node.py',
        name='particle_filter',
        parameters=[config],
        output='screen',
    )

    return LaunchDescription([particle_filter_node])
