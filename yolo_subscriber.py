# 以下は raw/compressed 両対応の YOLO 購読ノード（単体実行用）
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
import os

# トピック名（必要ならここを変更）
RAW_TOPIC = '/camera/camera_sensor/image_raw'
COMPRESSED_TOPIC = '/camera/camera_sensor/image_raw/compressed'

class YOLOSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # 初回は自動ダウンロード
        self.get_logger().info('YOLO model loaded')
        # まず raw を試みる（存在しなければ compressed を購読）
        try:
            self.sub = self.create_subscription(Image, RAW_TOPIC, self.cb_image, 10)
            self.get_logger().info(f'Subscribed to {RAW_TOPIC}')
        except Exception:
            # fallback: compressed
            self.sub = self.create_subscription(CompressedImage, COMPRESSED_TOPIC, self.cb_compressed, 10)
            self.get_logger().info(f'Subscribed to {COMPRESSED_TOPIC}')

    def cb_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return
        self.run_inference(cv_img)

    def cb_compressed(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'compressed decode error: {e}')
            return
        self.run_inference(cv_img)

    def run_inference(self, cv_img):
        # 画像が None のときは無視
        if cv_img is None:
            return
        results = self.model(cv_img, imgsz=416, conf=0.25)
        # 簡易出力（必要なら publish 等を追加）
        det_count = sum(1 for r in results if len(getattr(r, 'boxes', []))>0)
        self.get_logger().info(f'Inference done, results objects: {det_count}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
