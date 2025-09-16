import cv2
import base64
import time
from datetime import datetime
import config
from common import CacheHelper
from inference import TRT_YOLOv8

class Producer:
    def __init__(self, camera_id,engine_path=config.ENGINE_PATH):
        self.engine = TRT_YOLOv8(engine_path)
        self.cap = cv2.VideoCapture(config.STREAM_URL)
        self.camera_id = camera_id
        self.cache = CacheHelper()
        self.now = datetime.now()

        self.input_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.target_fps = config.TARGET_FPS
        self.skip_ratio = int(round(self.input_fps / self.target_fps)) if self.input_fps > self.target_fps else 1

    # def run_yolo_detection(self, frame):
    #     results = self.engine.predict(frame, conf=config.DETECTION_CONFIDENCE, imgsz=config.DETECTION_IMAGE_SIZE, device=0)
    #     detections = []
    #     for result in results:
    #         if result.boxes is not None:
    #             for box in result.boxes:
    #                 cls_id = int(box.cls[0].item())
    #                 conf = float(box.conf[0].item())
    #                 xyxy = box.xyxy[0].tolist()
    #                 if float(conf) > 0.2 and cls_id == 0:
    #                     detections.append({
    #                         "class_id": cls_id,
    #                         "confidence": conf,
    #                         "bbox": xyxy,
    #                         "class_name": self.engine.names[cls_id]
    #                     })
    #     return detections

    def encode_frame(self, frame, detections):
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return None
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        return {"frame": frame_b64, "detections": detections}

    def send_to_redis(self, frame_id, data):
        if frame_id == 0:
            self.now = datetime.now()
        timestamp = self.now.strftime("%S")

        
        key = f"cam{self.camera_id}_{timestamp}_{frame_id}"
        message = {key: data}
        self.cache.set_json(message)
        print(f"[Producer] Stored frame {frame_id} with key {key}")

    def run(self):
        frame_id = 0
        raw_frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            raw_frame_count += 1
            if raw_frame_count % self.skip_ratio != 0:
                continue

            frame = cv2.resize(frame, config.DETECTION_IMAGE_SIZE)
            # detections = self.run_yolo_detection(frame)
            st =time.time()
            detections = self.engine.detect(frame)
            print(detections)
            print("Detection time:",time.time()-st)
            
            encoded_data = self.encode_frame(frame, detections)
            if encoded_data:
                self.send_to_redis(frame_id % 15, encoded_data)

            frame_id += 1
            print("Total time:",time.time()-st)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    producer = Producer(config.CAMERA_ID)
    producer.run()
