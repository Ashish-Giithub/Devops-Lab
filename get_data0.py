import cv2
import base64
import numpy as np
from common import CacheHelper
from datetime import datetime
import config

class Consumer:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.now = datetime.now()
        self.cache = CacheHelper()

    def get_from_redis(self, frame_id):
        if frame_id == 0:
            self.now = datetime.now()
        timestamp = self.now.strftime("%S")
        

        key = f"cam{self.camera_id}_{timestamp}_{frame_id}"
        data = self.cache.get_json(key)
        if data:
            print(f"[Consumer] Retrieved {key}")
            # print(data)
            return data
        return None

    def draw_detections(self, data):
        frame_b64 = data["frame"]
        detections = data["detections"]

        frame_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["confidence"]
            # name = det["class_name"]
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f"{name}:{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        return frame

    def run(self):
        frame_id = 0
        while True:
            data = self.get_from_redis(frame_id % 15)
            if data:
                # print("Detections: ", data["detections"])
                frame = self.draw_detections(data)
                cv2.imshow(f"Camera {self.camera_id}", frame)

            frame_id += 1
            # if cv2.waitKey(100) & 0xFF == ord('q'):
            #     break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    consumer = Consumer(config.CAMERA_ID)
    consumer.run()
