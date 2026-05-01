from setuptools import setup
import os
from glob import glob

package_name = 'particle_filter'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Neel Shejwalkar',
    maintainer_email='nshej@seas.upenn.edu',
    description='MCL particle filter localization for F1Tenth',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'particle_filter_node = particle_filter.particle_filter_node:main',
        ],
    },
)
