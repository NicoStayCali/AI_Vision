import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import numpy as np
from cv_bridge import CvBridge
import cv2
import pcl
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import math

class DepthCloud(Node):

    def __init__(self):
        super().__init__('image_sub_py')
        self.subscriber_box = self.create_subscription(Float32MultiArray, '/AI/box', self.box_callback, 1)
        self.subscriber_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 1)
        #self.subscriber_depth = self.create_subscription(Image, '/AI/crop', self.depth_callback, 1)
        self.publisher_cloud = self.create_publisher(PointCloud2, '/AI/point_cloud', 10)

        self.box_data = []
        self.br = CvBridge()
        self.br_pub = CvBridge()
        timer_period = 0.5  # seconds
        self.i = 0.0
        self.timer_ = self.create_timer(timer_period, self.depth2PointCloud)
        self.current_depth = np.zeros((480, 640, 3), np.uint8)
        self.subscriber_box
        self.subscriber_depth

    def box_callback(self, msg):
        self.box_data = msg.data.tolist()

    def depth_callback(self, data):
        self.current_depth = self.br.imgmsg_to_cv2(data,"32FC1")
        self.crop()
        point_cloud = self.depth2PointCloud(self.current_depth)
        cloud_msg = pc2.create_cloud_xyz32(data.header, point_cloud)
        self.publisher_cloud.publish(cloud_msg)
        
    def crop(self):
        if len(self.box_data) > 3:
           x1 = math.floor(self.box_data[0])
           y1 = math.floor(self.box_data[1])
           x2 = math.ceil(self.box_data[2])
           y2 = math.ceil(self.box_data[3])
           
           frame = self.current_depth
           mask = np.zeros(frame.shape[:2], np.uint8)
           mask[y1:y2, x1:x2] = 255
           self.current_depth = cv2.bitwise_and(frame, frame, mask=mask)    
        
    def depth2PointCloud(self, depth=None): #, rgb):
    	if depth is None:
    	   return
    	height_im = 480
    	weight_im = 640
    	intrinsics_ppx = 326.5632629394531
    	intrinsics_ppy = 250.20266723632812
    	intrinsics_fx = 606.1911010742188
    	intrinsics_fy = 606.2071533203125
    	depth_scale = 0.001
    	clip_distance_max = 3500
    	
    	depth = np.asanyarray(depth) * depth_scale # 1000 mm => 0.001 meters
    	# rgb = np.asanyarray(rgb)
    	rows,cols  = depth.shape
    	
    	c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    	r = r.astype(float)
    	c = c.astype(float)
    	
    	valid = (depth > 0) & (depth < clip_distance_max) #remove from the depth image all values above a given value (meters).
    	valid = np.ravel(valid)
    	z = depth
    	x =  z * (c - intrinsics_ppx) / intrinsics_fx
    	y =  z * (r - intrinsics_ppy) / intrinsics_fy
    	
    	z = np.ravel(z)[valid]
    	x = np.ravel(x)[valid]
    	y = np.ravel(y)[valid]
    	#r = np.ravel(rgb[:,:,0])[valid]
    	#g = np.ravel(rgb[:,:,1])[valid]
    	#b = np.ravel(rgb[:,:,2])[valid]
    	
    	pointsxyz = np.dstack((x, y, z)) #, r, g, b))
    	pointsxyz = pointsxyz.reshape(-1,3)
    	
    	self.i += float(0.1)
    	return pointsxyz

def main(args=None):
    rclpy.init(args=args)
    depth_pc = DepthCloud()
    rclpy.spin(depth_pc)
    depth_pc.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
