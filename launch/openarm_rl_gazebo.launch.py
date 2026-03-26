from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, TimerAction, DeclareLaunchArgument,
    SetEnvironmentVariable, OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import subprocess


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('brain2rl_openarm')
    controller_config = os.path.join(pkg_share, 'config', 'openarm_controllers.yaml')

    # ---- use our custom sim URDF (has world link + ros2_control + Gazebo plugin) ----
    model_path = os.path.join(pkg_share, 'urdf', 'openarm_sim.urdf.xacro')
    urdf_xml = subprocess.check_output(['xacro', model_path]).decode('utf-8')

    run_policy   = LaunchConfiguration('run_policy')
    train_ros    = LaunchConfiguration('train_ros')
    run_agent    = LaunchConfiguration('run_agent')
    record_traj  = LaunchConfiguration('record_traj')
    policy_path  = LaunchConfiguration('policy_path')
    agent_ckpt   = LaunchConfiguration('agent_ckpt')
    reppo_train  = LaunchConfiguration('reppo_train')
    eeg_train    = LaunchConfiguration('eeg_train')
    load_run     = LaunchConfiguration('load_run')
    ckpt_path    = LaunchConfiguration('ckpt_path')

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf_xml, 'use_sim_time': True}],
    )

    # spawn at table surface (z=0.75); world→base_stand joint provides further offset
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'openarm',
            '-x', '0.0', '-y', '0.0', '-z', '0.75',
        ],
        output='screen',
    )

    jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--param-file', controller_config,
        ],
        output='screen',
    )

    jpos = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_group_position_controller',
            '--controller-manager', '/controller_manager',
            '--param-file', controller_config,
        ],
        output='screen',
    )

    spawn_cup = Node(
        package='brain2rl_openarm',
        executable='spawn_random_cup',
        output='screen',
    )

    policy_runner = Node(
        package='brain2rl_openarm',
        executable='policy_runner_node',
        output='screen',
        parameters=[{
            'policy_path': policy_path,
            'joint_state_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'control_rate_hz': 20.0,
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'action_mode': 'delta_position',
            'max_delta_rad': 0.05,
            'use_sim_time': True,
        }],
        condition=IfCondition(run_policy),
    )

    trainer = Node(
        package='brain2rl_openarm',
        executable='rl_train_joint_node',
        output='screen',
        parameters=[{
            'joint_states_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'control_rate_hz': 20.0,
            'episodes': 5,
            'steps_per_episode': 300,
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'max_delta_rad': 0.05,
            'save_path': 'ros_joint_policy.pt',
            'use_sim_time': True,
        }],
        condition=IfCondition(train_ros),
    )

    agent_runner = Node(
        package='brain2rl_openarm',
        executable='run_with_agent_node',
        output='screen',
        parameters=[{
            'agent_ckpt': agent_ckpt,
            'joint_states_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'control_rate_hz': 20.0,
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'max_delta_rad': 0.05,
            'instruction': 'pick up the cup',
            'use_sim_time': True,
        }],
        condition=IfCondition(run_agent),
    )

    traj_recorder = Node(
        package='brain2rl_openarm',
        executable='traj_recorder_node',
        output='screen',
        parameters=[{
            'out_dir': 'ros_trajs',
            'joint_states_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'use_sim_time': True,
        }],
        condition=IfCondition(record_traj),
    )

    # ── NEW: pure REPPO training from scratch in Gazebo ─────────
    reppo_trainer = Node(
        package='brain2rl_openarm',
        executable='reppo_train_node',
        output='screen',
        parameters=[{
            'joint_states_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'total_steps': 200000,
            'num_step': 128,
            'save_path': 'output/reppo_ros/checkpoint.pth',
            'use_sim_time': True,
        }],
        condition=IfCondition(reppo_train),
    )

    # ── NEW: EEG-conditioned REPPO training in Gazebo ────────────
    eeg_trainer = Node(
        package='brain2rl_openarm',
        executable='eeg_reppo_train_node',
        output='screen',
        parameters=[{
            'joint_states_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'total_steps': 200000,
            'num_step': 128,
            'eeg_token_pool': '',
            'save_path': 'output/eeg_reppo_ros/checkpoint.pth',
            'use_sim_time': True,
        }],
        condition=IfCondition(eeg_train),
    )

    # ── NEW: Load checkpoint (MuJoCo/ManiSkill) and run inference ─
    loader = Node(
        package='brain2rl_openarm',
        executable='load_and_run_node',
        output='screen',
        parameters=[{
            'ckpt_path': ckpt_path,
            'joint_states_topic': '/joint_states',
            'command_topic': '/joint_group_position_controller/commands',
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            'max_delta_rad': 0.05,
            'use_sim_time': True,
        }],
        condition=IfCondition(load_run),
    )

    return [
        rsp,
        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=5.0, actions=[jsb]),
        TimerAction(period=7.0, actions=[jpos]),
        TimerAction(period=9.0, actions=[spawn_cup]),
        TimerAction(period=10.0, actions=[
            policy_runner, trainer, agent_runner, traj_recorder,
            reppo_trainer, eeg_trainer, loader,
        ]),
    ]


def generate_launch_description():
    pkg_share    = get_package_share_directory('brain2rl_openarm')
    desc_share   = get_package_share_directory('openarm_description')
    world_file   = os.path.join(pkg_share, 'worlds', 'openarm_table.world')
    gazebo_model_path = os.path.join(desc_share, os.pardir)

    gui = LaunchConfiguration('gui')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py',
            )
        ),
        launch_arguments={
            'gui': gui,
            'verbose': 'true',
            'world': world_file,
        }.items(),
    )

    return LaunchDescription([
        SetEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=[
                os.environ.get('GAZEBO_MODEL_PATH', ''),
                ':',
                gazebo_model_path,
            ],
        ),

        DeclareLaunchArgument('gui',          default_value='true'),
        DeclareLaunchArgument('run_policy',   default_value='false'),
        DeclareLaunchArgument('train_ros',    default_value='false'),
        DeclareLaunchArgument('run_agent',    default_value='false'),
        DeclareLaunchArgument('record_traj',  default_value='false'),
        DeclareLaunchArgument('policy_path',  default_value='policy.pt'),
        DeclareLaunchArgument('agent_ckpt',   default_value=''),
        DeclareLaunchArgument('reppo_train',  default_value='false'),
        DeclareLaunchArgument('eeg_train',    default_value='false'),
        DeclareLaunchArgument('load_run',     default_value='false'),
        DeclareLaunchArgument('ckpt_path',    default_value=''),

        gazebo,
        OpaqueFunction(function=_launch_setup),
    ])
