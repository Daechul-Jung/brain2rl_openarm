from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    gui = LaunchConfiguration('gui')
    model = LaunchConfiguration('model')

    run_policy = LaunchConfiguration('run_policy')
    train_ros = LaunchConfiguration('train_ros')
    run_agent = LaunchConfiguration('run_agent')

    policy_path = LaunchConfiguration('run_agent')
    agent_ckpt = LaunchConfiguration('agent_ckpt')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')), 
            launch_arguments={'gui': gui, 'verbose': 'true', 'extra_gazebo_args': '-s libgazebo_ros_path_plugin.so'}.items()
        )
    
    robot_description = ParameterValue(
        Command(['xacro', model, ' use_sim:=true ros2_control:=true']),
        value_type = str
    )

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               output = 'screen', parameters=[{'robot_description': robot_description}])

    spawn_robot = Node(package='gazebo_ros', executable='spawn_entity.py',
                       arguments = ['-topic', 'robot_description', '-entity', 'openarm'],
                       output = 'screen')
    
    jsb = Node(package='controller_manager', executable='spawner',
               arguments = ['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
               output = 'screen')
               
    jpos = Node(package='cnotroller_manager', executable='spawner',
                arguments=['joint_group_position_controller', '--controller-manager', '/controller_manager'],
                output='screen')
    
    spawn_cup = Node(package='brain2rl_openarm', executable='spawn_random_cup', output = 'screen')

    policy_runner = Node(
        package = 'brain2rl_openarm',
        executable='policy_runner_node',
        output = 'screen',
        parameters=[{
            'policy_path': policy_path,
            'joint_state_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'control_rate_hz': 20.0,
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'action_mode': 'delta_position',
            'max_delta_rad': 0.05
        }],
        condition = IfCondition(train_ros)
    )

    trainer = Node(
        package='brain2rl_openarm',
        executable='rl_train_joint_node',
        output='screen',
        parameters = [{
            'joint_states_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'control_rate_hz': 20.0,
            'episodes': 5,
            'steps_per_episode': 300,
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'max_delta_rad': 0.05,
            'save_path': 'ros_joint_policy.pt'
        }],
        condition=IfCondition
    )

    agent_runner = Node(
        package='brain2rl_openarm',
        executable='run_with_agent_node',
        output = 'screen',
        parameters=[{
            'agent_ckpt': agent_ckpt,
            'joint_states_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'control_rate_hz': 20.0,
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'max_delta_rad': 0.05,
            'instruction': 'pick up the cup'
        }],
        condition= IfCondition(run_agent)
    )

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('model', default_value=os.path.join(
            get_package_share_directory('openarm_description'),
            'urdf', 'robot', 'v10.urdf.xacro'
        )),

        DeclareLaunchArgument('run_policy', default_value='false'),
        DeclareLaunchArgument('train_ros', default_value='false'),
        DeclareLaunchArgument('run_agent', default_value='false'),
        DeclareLaunchArgument('policy_path', default_value='policy.pt'),
        DeclareLaunchArgument('agent_ckpt', default_value=''),

        gazebo, rsp,

        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=4.0, actions=[jsb]),
        TimerAction(period=5.5, actions=[jpos]),
        TimerAction(period=6.5, actions=[spawn_cup]),
        TimerAction(period=7.0, actions=[policy_runner, trainer, agent_runner]),
    ])