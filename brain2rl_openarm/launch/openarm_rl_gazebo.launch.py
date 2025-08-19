from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    gui   = LaunchConfiguration('gui')
    model = LaunchConfiguration('model') 
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'gui': gui}.items()
    )

    robot_description = ParameterValue(
        Command(['xacro ', model, ' use_sim:=true']),
        value_type=str
    )

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               output='screen', parameters=[{'robot_description': robot_description}])

    spawn_robot = Node(package='gazebo_ros', executable='spawn_entity.py',
                       arguments=['-topic', 'robot_description', '-entity', 'openarm'],
                       output='screen')

    jsb  = Node(package='controller_manager', executable='spawner',
                arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
                output='screen')

    jpos = Node(package='controller_manager', executable='spawner',
                arguments=['joint_group_position_controller', '--controller-manager', '/controller_manager'],
                output='screen')

    spawn_cup = Node(package='brain2rl_openarm', executable='spawn_random_cup', output='screen')
    trainer   = Node(package='brain2rl_openarm', executable='rl_train_joint',
                     output='screen', arguments=['--episodes','5','--steps','300'])

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false'),
        # default can be your preferred file; you can also always override on CLI
        DeclareLaunchArgument('model', default_value='/home/jdcjdb/ros2_ws/src/openarm_description/urdf/robot/v10.urdf.xacro'),
        gazebo, rsp,
        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=4.0, actions=[jsb]),
        TimerAction(period=5.5, actions=[jpos]),
        TimerAction(period=7.0, actions=[spawn_cup]),
        TimerAction(period=9.0, actions=[trainer]),   # comment out if training from ML repo
    ])
