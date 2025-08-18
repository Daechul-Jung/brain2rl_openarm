# ros2_ws/src/brain2rl_openarm/brain2rl_openarm/launch/openarm_rl_gazebo.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    gui = LaunchConfiguration('gui')

    xacro_file = os.path.join(
        get_package_share_directory('openarm_description'),
        'urdf', 'openarm.urdf.xacro'   # change if your file has a different name
    )

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' use_sim:=true']),
        value_type=str
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'gui': gui}.items()
    )

    rsp = Node(package='robot_state_publisher',
               executable='robot_state_publisher',
               output='screen',
               parameters=[{'robot_description': robot_description}])

    spawn_robot = Node(package='gazebo_ros', executable='spawn_entity.py',
                       arguments=['-topic', 'robot_description', '-entity', 'openarm'],
                       output='screen')

    # Controller names must match your URDF ros2_control block
    jsb = Node(package='controller_manager', executable='spawner',
               arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
               output='screen')

    jpos = Node(package='controller_manager', executable='spawner',
                arguments=['joint_group_position_controller', '--controller-manager', '/controller_manager'],
                output='screen')

    spawn_cup = Node(package='brain2rl_openarm', executable='spawn_random_cup.py', output='screen')
    trainer   = Node(package='brain2rl_openarm', executable='rl_train_joint.py',
                     output='screen', arguments=['--episodes','5','--steps','300'])

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false'),
        gazebo,
        rsp,
        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=4.0, actions=[jsb]),
        TimerAction(period=5.5, actions=[jpos]),
        TimerAction(period=7.0, actions=[spawn_cup]),
        # comment this out if you’ll train from brain2rl/scripts instead
        TimerAction(period=9.0, actions=[trainer]),
    ])
