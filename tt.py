import cv2
import base64
import numpy as np
import time
from datetime import datetime
from common import CacheHelper
import os

# Ensure it's float, not string
MAX_FRAME_TIME = float(os.getenv("REDIS_MAX_FRAME_TIME", 0.03))


class Consumer:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cache = CacheHelper()
        # Start from current second
        self.current_second = int(datetime.now().strftime("%S"))
        self.frame_id = 0
        self.current_frame_key = None
        self.last_frame = None
        print(f"[Consumer] Initialized for camera {camera_id}")

    def get_from_redis(self):
        """Fetch latest frame+detections from Redis"""
        latest_pointer_key = f"cam{self.camera_id}_latest"
        print(f"[Consumer] Checking for key: {latest_pointer_key}")
        
        # Get the latest key pointer
        latest_key_data = self.cache.get_json(latest_pointer_key)
        
        if latest_key_data is None:
            print(f"[Consumer] No data found for {latest_pointer_key}")
            return None
            
        # Handle different data structures
        if isinstance(latest_key_data, str):
            latest_key = latest_key_data
        elif isinstance(latest_key_data, dict) and latest_pointer_key in latest_key_data:
            latest_key = latest_key_data[latest_pointer_key]
        else:
            print(f"[Consumer] Unexpected data structure: {latest_key_data}")
            return None
            
        print(f"[Consumer] Latest key: {latest_key}")
        
        # Check if this is a new frame
        if latest_key and latest_key != self.current_frame_key:
            # Get the actual frame data
            raw_data = self.cache.get_json(latest_key)
            
            if raw_data:
                # Handle the nested structure from producer
                if isinstance(raw_data, dict) and latest_key in raw_data:
                    data = raw_data[latest_key]
                elif isinstance(raw_data, dict) and "frame" in raw_data:
                    data = raw_data
                else:
                    print(f"[Consumer] Unexpected frame data structure: {type(raw_data)}")
                    return None
                    
                self.current_frame_key = latest_key
                print(f"[Consumer] Retrieved new frame: {latest_key}")
                return data
            else:
                print(f"[Consumer] No data found for frame key: {latest_key}")
                
        return None

    def draw_detections(self, data):
        """Decode base64 frame + return detections"""
        try:
            frame_b64 = data["frame"]
            detections = data.get("detections", [])

            frame_bytes = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("[Consumer] Failed to decode frame")
                return None, []
                
            return frame, detections
        except Exception as e:
            print(f"[Consumer] Error decoding frame: {e}")
            return None, []

    def run(self):
        """Fetches the latest frame (or reuses old one if no update)"""
        data = self.get_from_redis()

        if data:
            frame, detections = self.draw_detections(data)
            if frame is not None:
                self.last_frame = frame
                return frame, detections
        
        # Fallback to last frame if available
        if self.last_frame is not None:
            print("[Consumer] Reusing last frame")
            return self.last_frame, []
        else:
            print("[Consumer] No frame available")
            return None, []

    # def check_available_keys(self):
    #     """Debug method to see what keys are available in Redis"""
    #     try:
    #         # Use Redis KEYS command to see what's available (only for debugging)
    #         import redis
    #         r = redis.StrictRedis(host="localhost", port=6379, password="moksa123", decode_responses=True)
    #         keys = r.keys(f"cam{self.camera_id}*")
    #         print(f"[Consumer] Available keys for camera {self.camera_id}: {keys}")
            
    #         # Also check for any keys that might match the pattern
    #         all_keys = r.keys("*")
    #         matching_keys = [k for k in all_keys if str(self.camera_id) in k]
    #         print(f"[Consumer] All matching keys: {matching_keys}")
            
    #     except Exception as e:
    #         print(f"[Consumer] Error checking keys: {e}")


if __name__ == "__main__":
    camera_id = 2272  # Make sure this matches what your producer is using
    consumer = Consumer(camera_id=camera_id)
    
    # Debug: Check what keys are available
    # consumer.check_available_keys()
    
    print(f"[Consumer] Starting consumer for camera {camera_id}")
    frame_count = 0

    while True:
        start_time = time.time()
        
        frame, detections = consumer.run()
        if frame is not None:
            frame_count += 1
            print(f"[Consumer] Processing frame {frame_count} with {len(detections)} detections")
            
            # Draw detections if needed
            for det in detections:
                try:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, det.get("class_name", "person"), 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"[Consumer] Error drawing detection: {e}")

            cv2.imshow("Consumer View", frame)
        else:
            print("[Consumer] Waiting for frames...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Maintain frame rate
        elapsed = time.time() - start_time
        if elapsed < MAX_FRAME_TIME:
            time.sleep(MAX_FRAME_TIME - elapsed)

    cv2.destroyAllWindows()