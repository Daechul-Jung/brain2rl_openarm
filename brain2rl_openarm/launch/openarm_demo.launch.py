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

# ros2_ws/src/brain2rl_openarm/brain2rl_openarm/launch/openarm_demo.launch.py
# from launch import LaunchDescription
# from launch.actions import IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from ament_index_python.packages import get_package_share_directory
# import os

# def generate_launch_description():
#     moveit_pkg = 'openarm_bimanual_moveit_config'
#     demo_launch_path = os.path.join(
#         get_package_share_directory(moveit_pkg), 'launch', 'demo.launch.py'
#     )
#     moveit_demo = IncludeLaunchDescription(PythonLaunchDescriptionSource(demo_launch_path))
#     return LaunchDescription([moveit_demo])
