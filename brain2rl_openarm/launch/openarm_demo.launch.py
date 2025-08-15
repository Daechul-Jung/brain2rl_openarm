from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription      
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('openarm_bimanual_moveit_config')
    demo = os.path.join(pkg, 'launch', 'demo.launch.py')
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(demo))
    ])
