import cv2
import base64
import time
from datetime import datetime
from threading import Thread, Lock
from ultralytics import YOLO
import config
from common import CacheHelper   # <-- Redis helper

class MultiCameraPoseDetection:
    def __init__(self):
        self.model_path = config.MODEL_PATH
        self.model = YOLO(self.model_path)  # Ensure it's a pose-trained model
        self.model_lock = Lock()  # Lock for shared model inference

        # Redis client
        self.cache = CacheHelper()

        # Initialize camera captures
        self.cameras = {}
        self.frame_ids = {}
        self.skip_ratios = {}

        for cam in config.CAMERA_STREAMS:
            cam_id = cam["id"]
            url = cam["url"]
            cap = cv2.VideoCapture(url)
            self.cameras[cam_id] = cap
            self.frame_ids[cam_id] = 0

            input_fps = cap.get(cv2.CAP_PROP_FPS)
            if input_fps == 0 or input_fps != input_fps:
                input_fps = 30
            target_fps = config.TARGET_FPS
            skip_ratio = int(round(input_fps / target_fps)) if input_fps > target_fps else 1
            self.skip_ratios[cam_id] = skip_ratio

            print(f"[INFO] Camera {cam_id}: Input FPS={input_fps}, Target FPS={target_fps}, Skip ratio={skip_ratio}")

    def run_yolo_pose(self, frame):
        """Run YOLO Pose inference with thread lock"""
        if frame is None:
            return []

        with self.model_lock:
            results = self.model.predict(
                frame,
                task="pose",
                conf=config.DETECTION_CONFIDENCE,
                imgsz=config.DETECTION_IMAGE_SIZE,
                device=0
            )

        poses = []
        for result in results:
            if result.keypoints is not None:
                # result.keypoints.data is a tensor of shape [num_detections, num_keypoints, 3]
                if result.keypoints is not None:
                    keypoints_tensor = result.keypoints.data.cpu().numpy()  # convert to numpy
                    for i, keypoints in enumerate(keypoints_tensor):  # iterate over detections
                        keypoints_list = keypoints.tolist()
                        poses.append({
                            "keypoints": keypoints_list,
                            "class_id": int(result.boxes.cls[i].item()) if result.boxes is not None else -1,
                            "confidence": float(result.boxes.conf[i].item()) if result.boxes is not None else 1.0
                        })
        return poses

    def encode_frame(self, frame, poses):
        if frame is None:
            return None
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return None
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        return {"frame": frame_b64, "poses": poses}

    def send_to_redis(self, cam_id, frame_id, data):
        now = datetime.now()
        timestamp = now.strftime("%S")
        key = f"cam{cam_id}_{timestamp}_{frame_id}"
        message = {key: {"frame": data["frame"], "poses": data["poses"]}}
        self.cache.set_json(message)

    def process_camera(self, cam_id, cap):
        raw_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {cam_id}: Failed to read frame")
                break

            raw_frame_count += 1
            if raw_frame_count % self.skip_ratios[cam_id] != 0:
                continue

            frame = cv2.resize(frame, config.DETECTION_IMAGE_SIZE)

            st = time.time() * 1000
            poses = self.run_yolo_pose(frame)
            print(poses)
            print(f"[Camera {cam_id}] Pose detection time (ms): {time.time()*1000 - st}")

            mt = time.time() * 1000
            encoded_data = self.encode_frame(frame, poses)
            if encoded_data is None:
                continue

            frame_id = self.frame_ids[cam_id] % 15
            self.send_to_redis(cam_id, frame_id, encoded_data)
            print(f"[Camera {cam_id}] Published frame {frame_id} to Redis. Time(ms): {time.time()*1000 - mt}")

            self.frame_ids[cam_id] += 1

    def run(self):
        threads = []
        for cam_id, cap in self.cameras.items():
            t = Thread(target=self.process_camera, args=(cam_id, cap), daemon=True)
            t.start()
            threads.append(t)

        print("[INFO] Multi-camera pose detection started. Press Ctrl+C to exit.")
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("[INFO] Stopping all cameras...")

        for cap in self.cameras.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = MultiCameraPoseDetection()
    detector.run()
