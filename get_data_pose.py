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

        # Define skeleton connections (COCO-style for 17 keypoints)
        self.skeleton = [
            (5, 7), (7, 9),        # Left arm
            (6, 8), (8, 10),       # Right arm
            (5, 6),                # Shoulders
            (11, 13), (13, 15),    # Left leg
            (12, 14), (14, 16),    # Right leg
            (11, 12),              # Hips
            (5, 11), (6, 12)       # Torso connections
        ]

    def get_from_redis(self, timestamp, frame_id):
        # if frame_id == 0:
        #     self.now = datetime.now()
        self.now = datetime.now()
        
        key = f"cam{self.camera_id}_{timestamp}_{frame_id}"
        data = self.cache.get_json(key)
        if data:
            print(f"[Consumer] Retrieved {key}")
            return data
        return None

    def draw_poses(self, data):
        frame_b64 = data["frame"]
        poses = data["poses"]

        # Decode frame from base64
        frame_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        for pose in poses:
            keypoints = pose.get("keypoints", [])
            conf = pose.get("confidence", 0)

            # Draw keypoints
            for idx, kp in enumerate(keypoints):
                x, y, score = kp
                if score > 0.3:  # draw only confident points
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Draw skeleton
            for (i, j) in self.skeleton:
                if i < len(keypoints) and j < len(keypoints):
                    xi, yi, si = keypoints[i]
                    xj, yj, sj = keypoints[j]
                    if si > 0.3 and sj > 0.3:
                        cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (255, 0, 0), 2)

            # Optionally display confidence score
            if len(keypoints) > 0:
                x0, y0, _ = keypoints[0]
                cv2.putText(frame, f"Conf:{conf:.2f}", (int(x0), int(y0) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame

    def run(self):
        frame_id = 0
        timestamp = datetime.now().strftime("%S")
        while True:
            data = self.get_from_redis(timestamp, frame_id % 15)
            if data:
                frame = self.draw_poses(data)
                cv2.imshow(f"Camera {self.camera_id}", frame)

            frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    consumer = Consumer(config.CAMERA_ID)
    consumer.run()
