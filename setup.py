from setuptools import find_packages, setup

package_name = 'brain2rl_openarm'

from setuptools import setup
package_name = 'brain2rl_openarm'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name + '/launch', ['brain2rl_openarm/launch/openarm_demo.launch.py']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daechul Jung',
    maintainer_email='jungdaechul@berkeley.edu',
    description='OpenArm sim wrapper for brain2rl',
    license='MIT',
    entry_points={'console_scripts': []},
)
