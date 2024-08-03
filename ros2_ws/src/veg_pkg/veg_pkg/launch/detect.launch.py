from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
       Node(namespace= "rqt", package='rqt_gui', executable='rqt_gui'),
       Node(namespace= "detected", package='veg_pkg', executable='ai_id')
    ])
