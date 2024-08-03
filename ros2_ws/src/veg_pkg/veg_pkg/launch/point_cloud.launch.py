from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
       Node(namespace= "point_cloud", package='veg_pkg', executable='point_cloud'),
       Node(namespace= "rviz2", package='rviz2', executable='rviz2', arguments=['-d', 'veg_pkg/config/rviz2_point_cloud.rviz']),
       Node(namespace= "tf_map2cam", package='tf2_ros', executable='static_transform_publisher', arguments=['0', '0', '0', '0', '3.14', '0', 'camera_depth_optical_frame', 'map'])
    ])
