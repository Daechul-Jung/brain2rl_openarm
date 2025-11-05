from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'brain2rl_openarm'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name + '/launch', 
         ['brain2rl_openarm/launch/openarm_demo.launch.py']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob(os.path.join(package_name, 'launch', '*.launch.py'))),
        # install config files (optional, but useful)
        ('share/' + package_name + '/config',
            glob(os.path.join(package_name, 'config', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daechul Jung',
    maintainer_email='jungdaechul@berkeley.edu',
    description='OpenArm sim wrapper for brain2rl',
    license='MIT',
    entry_points={
        'console_scripts': [
            ####  Added for more nodes
            'rl_train_joint = brain2rl_openarm.scripts.rl_train_joint:main',
            'run_with_agent = brain2rl_openarm.scripts.run_with_agent:main',
            'spawn_random_cup = brain2rl_openarm.scripts.spawn_random_cup:main',
            'test_env = brain2rl_openarm.scripts.test_env:main',
            'train_ppo_openarm = brain2rl_openarm.scripts.train_ppo_openarm:main',
        ],
    },
)
