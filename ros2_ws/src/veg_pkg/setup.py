from setuptools import find_packages, setup

package_name = 'veg_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['veg_pkg/launch/detect.launch.py']),
        ('share/' + package_name, ['veg_pkg/launch/point_cloud.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nicola',
    maintainer_email='nico.staycali@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ # 'ex_pub_py = veg_pkg.ex_pub_py:main',
                             # 'ex_sub_py = veg_pkg.ex_sub_py:main',
                             # 'image_sub = veg_pkg.image_sub:main',
                             'ai_id = veg_pkg.ai_id:main',
                             'point_cloud = veg_pkg.point_cloud:main',
                             # 'ros2bag_video = veg_pkg.ros2bag_video:main',
                             # 'align-depth2color = veg_pkg.align-depth2color:main'
                             'mask = veg_pkg.mask:main'
        ],
    },
)
