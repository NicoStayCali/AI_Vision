import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from PIL import Image as Image_pil
import numpy as np
import math


class ImageMaskNode(Node):

    def __init__(self):
        super().__init__('image_mask_node')
        self.subscriber_box = self.create_subscription(Float32MultiArray, '/AI/box', self.box_callback, 1)
        self.subscriber_image = self.create_subscription(Image, '/camera/camera/color/image_raw', self.listener_callback,1)
        self.publisher_image = self.create_publisher(Image, '/AI/crop', 10)
        
        self.box_data = []
        self.br = CvBridge()
        self.br_pub = CvBridge()
        timer_period = 0.5
        self.i = 0.0
        self.timer_ = self.create_timer(timer_period, self.crop)
        self.current_frame = np.zeros((480, 640, 3), np.uint8)
        self.subscriber_box
        self.subscriber_image
        
    def box_callback(self, msg):
        self.box_data = msg.data.tolist()
        
    def listener_callback(self, msg):
        self.current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    def crop(self):
        if len(self.box_data) > 3:
           x1 = math.floor(self.box_data[0])
           y1 = math.floor(self.box_data[1])
           x2 = math.ceil(self.box_data[2])
           y2 = math.ceil(self.box_data[3])
           
           frame = self.current_frame
           mask = np.zeros(frame.shape[:2], np.uint8)
           mask[y1:y2, x1:x2] = 255
           img_masked = cv2.bitwise_and(frame, frame, mask=mask)
           
           cropped_frame_img = np.zeros((480, 640, 3), np.uint8)
           cropped_frame_img = Image_pil.fromarray(img_masked)
           cropped_frame_cv2 = np.array(cropped_frame_img)
           image_pub = Image()
           image_pub = self.br_pub.cv2_to_imgmsg(cropped_frame_cv2, encoding="bgr8")
           self.publisher_image.publish(image_pub)

        self.i += float(0.1)
   
def main(args=None):
    rclpy.init(args=args)
    image_mask_node = ImageMaskNode()
    rclpy.spin(image_mask_node)
    image_mask_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
