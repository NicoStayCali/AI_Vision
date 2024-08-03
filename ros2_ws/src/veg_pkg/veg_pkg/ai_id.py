import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import numpy as np
from cv_bridge import CvBridge
import cv2
from PIL import Image as Image_pil
import yolov5

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_sub_py')
        self.subscriber_image = self.create_subscription(Image, '/camera/camera/color/image_raw', self.img_callback, 1)
        self.publisher_image = self.create_publisher(Image, '/AI/detection', 10)
        self.publisher_box_mask = self.create_publisher(Float32MultiArray, '/AI/box', 10)

        self.br = CvBridge()
        self.br_pub = CvBridge()
        timer_period = 0.5
        self.i = 0.0
        self.timer_ = self.create_timer(timer_period, self.yolo_detect_veg)
        self.current_frame = np.zeros((480, 640, 3), np.uint8)
        self.subscriber_image

    def img_callback(self, data):
        self.current_frame = self.br.imgmsg_to_cv2(data)
        
    def yolo_detect_veg(self):
        frame = self.current_frame
        model = yolov5.load('/home/klaster/AI_Vision/src/veg_pkg/veg_pkg/yolov5/iceberg.pt')
        results = model(frame)
        result_img = results.render()
        img_with_boxes = result_img[0]
        #results.show()
        #crops = results.crop(save=False)
        
        boxes = results.xyxy[0]
        if len(boxes) > 0:
           box = boxes[0, :4].tolist()
           box_msg = Float32MultiArray()
           box_msg.data = box
           self.publisher_box_mask.publish(box_msg)
        else:
           print("La lista 'boxes' è vuota.")
        
        '''
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            box_dimensions = [x1, y1, x2, y2]
            box_msg = Float32MultiArray()
            box_msg.data = box_dimensions
            self.publisher_box_mask.publish(box_msg)
        '''
        
        annotated_frame_img = np.zeros((480, 640, 3), np.uint8)
        annotated_frame_img = Image_pil.fromarray(img_with_boxes)
        annotated_frame_cv2 = np.array(annotated_frame_img)
        image_pub = Image()
        image_pub = self.br_pub.cv2_to_imgmsg(annotated_frame_cv2, encoding="rgb8")
        self.publisher_image.publish(image_pub)
        
        self.i += float(0.1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
