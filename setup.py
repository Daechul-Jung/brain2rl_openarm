from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'brain2rl_openarm'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daechul Jung',
    maintainer_email='jungdaechul@berkeley.edu',
    description='OpenArm sim wrapper for brain2rl',
    license='MIT',
    entry_points={
        'console_scripts': [
            'policy_runner_node = brain2rl_openarm.nodes.policy_runner_node:main',
            'rl_train_joint_node = brain2rl_openarm.nodes.rl_train_joint_node:main',
            'joint_order_node = brain2rl_openarm.nodes.joint_order_node:main',
            'ros_env_node = brain2rl_openarm.nodes.ros_env_node:main',
            'run_with_agent_node = brain2rl_openarm.nodes.run_with_agent_node:main',
            'traj_recorder_node = brain2rl_openarm.nodes.traj_recorder_node:main',
            'spawn_random_cup = brain2rl_openarm.scripts.spawn_random_cup:main',
            'reppo_train_node = brain2rl_openarm.nodes.reppo_train_node:main',
            'eeg_reppo_train_node = brain2rl_openarm.nodes.eeg_reppo_train_node:main',
            'load_and_run_node = brain2rl_openarm.nodes.load_and_run_node:main',
        ],
    },
)
