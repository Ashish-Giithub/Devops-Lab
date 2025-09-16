import cv2
import base64
import time
from datetime import datetime
from threading import Thread, Lock
from queue import Queue, Empty
from ultralytics import YOLO
# REMOVE THIS LINE: import pycuda.autoinit  # <-- This is causing the problem!
import config
import numpy as np
from antmedia1.webrtc_subscriber import AntMediaCamera
from common import CacheHelper
from camera_intialize import CameraInit
# from trt_engine import TRTEngine
 
MAX_FRAME_TIME = 0.030  # 30 ms per frame budget (~33 FPS)
 
class MultiCameraDetection:
    def __init__(self, engine_path=None, iou_thresh=0.45):
        self.cache = CacheHelper()
 
        self.cameras = {}
        self.frame_ids = {}
        self.skip_ratios = {}
        self.current_seconds = {}  # track seconds per camera
        self.last_outputs = {}  # store last frame+detections per camera
        
        # Single queue for all detection work
        self.detection_queue = Queue(maxsize=20)
        self.model = YOLO(config.MODEL_PATH)
        
        # Initialize TensorRT engine in the detection thread, not here
        self.engine_path = engine_path
        self.iou_thresh = iou_thresh
        self.engine = None  # Will be initialized in detection thread
        self.model_lock = Lock()
 
        camera_initializer = CameraInit()
 
        for cam in config.CAMERA_STREAMS:
            cam_id = cam["id"]
            client_type = cam.get("client_type", "webrtc")
            url = cam.get("url")
 
            cap = camera_initializer.camera_init(client_type=client_type, rtsp_url=url, camera_id=cam_id)
            if not cap:
                print(f"[ERROR] Failed to initialize camera {cam_id}. Skipping.")
                continue
 
            self.cameras[cam_id] = cap
            self.frame_ids[cam_id] = 0
            self.current_seconds[cam_id] = int(datetime.now().strftime("%S"))
            self.last_outputs[cam_id] = None
 
            if isinstance(cap, cv2.VideoCapture):
                input_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            else:
                input_fps = 15
            
            target_fps = config.TARGET_FPS
            skip_ratio = int(round(input_fps / target_fps)) if input_fps > target_fps else 1
            self.skip_ratios[cam_id] = skip_ratio
 
            print(f"[INFO] Camera {cam_id}: Client={client_type}, Input FPS={input_fps}, Target FPS={target_fps}, Skip ratio={skip_ratio}")
 
    def frame_reader(self, cam_id, cap):
        target_fps = config.TARGET_FPS
        frame_interval = 1.0 / target_fps

        last_time = time.perf_counter()
        frame_counter = 0
        fps_timer = last_time
        fps_window = 5.0  # seconds

        while True:
            # Read frame from AntMediaCamera or OpenCV VideoCapture
            if isinstance(cap, AntMediaCamera):
                ret, frame = cap.read()
                if not ret or not isinstance(frame, np.ndarray):
                    continue
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

            frame = cv2.resize(frame, config.DETECTION_IMAGE_SIZE)

            # Put frame in detection queue with camera ID
            try:
                self.detection_queue.put((cam_id, frame), timeout=0.001)
            except:
                # Drop frame if queue is full - prevents backup
                pass

            now = time.perf_counter()
            elapsed = now - last_time

            # Drift-compensated sleep
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Increment based on target interval (not actual now)
            last_time += frame_interval
            frame_counter += 1

            # FPS reporting (every 5 sec for stability)
            if now - fps_timer >= fps_window:
                actual_fps = frame_counter / (now - fps_timer)
                print(f"[Camera {cam_id}] Actual FPS: {actual_fps:.2f} (target {target_fps})")
                fps_timer = now
                frame_counter = 0

 
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
 
    def encode_frame(self, frame, detections):
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return None
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        return {"frame": frame_b64, "detections": detections}
 
    def send_to_redis(self, cam_id, frame_id, data):
        timestamp = self.current_seconds[cam_id]
        key = f"cam{cam_id}_{timestamp}_{frame_id}"
        print("---------------------------", key, "--------------------------------------")
        
        message = {key: {"frame": data["frame"], "detections": data["detections"]}}
        self.cache.set_json(message)
        
        # Set latest frame pointer
        latest_key = f"cam{cam_id}_latest"
        self.cache.set_json({latest_key: key})
        
        if frame_id == 14:
            self.current_seconds[cam_id] = (self.current_seconds[cam_id] + 1) % 60
 
    def single_detection_worker(self):
        """Single worker thread that processes frames from all cameras"""
        print("[INFO] Starting detection worker thread...")
        
        while True:
            try:
                cam_id, frame = self.detection_queue.get(timeout=1.0)
            except Empty:
                continue
                
            if frame is None:
                break
                
            # Check for valid frame
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                print("Skipping invalid frame")
                continue
                
            start_time = time.perf_counter()
            
            try:
                detections = self.run_yolo_pose(frame)
                duration = time.perf_counter() - start_time
                print(f"[Camera {cam_id}] Detection time: {duration*1000:.2f} ms")
 
                # If too slow, reuse previous result
                if duration > MAX_FRAME_TIME and self.last_outputs[cam_id]:
                    encoded_data = self.last_outputs[cam_id]
                    print(f"[Camera {cam_id}] Reusing previous result (detection too slow)")
                else:
                    encoded_data = self.encode_frame(frame, detections)
                    if encoded_data:
                        self.last_outputs[cam_id] = encoded_data
 
                if encoded_data is None:
                    continue
 
                frame_id = self.frame_ids[cam_id] % 15
                self.send_to_redis(cam_id, frame_id, encoded_data)
                print(f"[Camera {cam_id}] Published frame {frame_id} to Redis.")
                self.frame_ids[cam_id] += 1
 
                # Ensure fixed output cadence
                elapsed = time.perf_counter() - start_time
                if elapsed < MAX_FRAME_TIME:
                    time.sleep(MAX_FRAME_TIME - elapsed)
                    
            except Exception as e:
                print(f"[ERROR] Detection failed for camera {cam_id}: {e}")
                continue
 
    def run(self):
        threads = []
        
        # Start frame reader threads
        for cam_id, cap in self.cameras.items():
            t = Thread(target=self.frame_reader, args=(cam_id, cap), daemon=True)
            t.start()
            threads.append(t)
            print(f"[INFO] Started frame reader for camera {cam_id}")
        
        # Start single detection worker
        detection_thread = Thread(target=self.single_detection_worker, daemon=True)
        detection_thread.start()
        threads.append(detection_thread)
        print("[INFO] Started detection worker")
 
        print("[INFO] Multi-camera detection started. Press Ctrl+C to exit.")
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("[INFO] Stopping all cameras...")
 
        # Cleanup
        for cap in self.cameras.values():
            if isinstance(cap, cv2.VideoCapture):
                cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = MultiCameraDetection(engine_path="yolo11m.engine", iou_thresh=0.45)
        detector.run()
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received. Stopping gracefully...")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()