from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Gazebo
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        )
    )

    # OpenArm bringup (controllers, robot description)
    bringup_pkg = 'openarm_bringup'
    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(bringup_pkg), 'launch', 'bringup.launch.py')
        )
    )

    # Our scripts
    spawn_cup = Node(
        package='brain2rl_openarm',
        executable='spawn_random_cup.py',
        name='spawn_random_cup',
        output='screen'
    )

    rl_trainer = Node(
        package='brain2rl_openarm',
        executable='rl_train_joint.py',
        name='rl_train_joint',
        output='screen',
        arguments=['--episodes', '5', '--steps', '300']
    )

    # Stagger spawns slightly so Gazebo and controllers are up
    return LaunchDescription([
        gazebo_launch,
        bringup_launch,
        TimerAction(period=4.0, actions=[spawn_cup]),
        TimerAction(period=6.0, actions=[rl_trainer]),
    ])
