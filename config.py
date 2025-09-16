import os

MODEL_PATH = str(os.getenv('MODEL_PATH',"yolo11m-pose.pt"))
print(MODEL_PATH)
CAMERA_ID = int(os.getenv('CAMERA_ID','1'))
print(CAMERA_ID)
STREAM_URL = str(os.getenv('CAMERA_URL_1',"rtsp://admin:Sethi19000@97.94.217.18:554/Streaming/channels/401"))
print(STREAM_URL)
STREAM_URL_2 = str(os.getenv('CAMERA_URL_2',"rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/802"))
print(STREAM_URL_2)
ENGINE_PATH = str(os.getenv('MODEL_PATH',"yolo11m.engine"))
print(ENGINE_PATH)
DETECTION_CONFIDENCE = float(os.getenv('MODEL_PATH',0.5))
print(DETECTION_CONFIDENCE)
DETECTION_IMAGE_SIZE = os.getenv('DETECTION_IMAGE_SIZE',(640,480))
EXPIRY_TIME = int(os.getenv('EXPIRY_TIME',10))
print(EXPIRY_TIME)
TARGET_FPS = int(os.getenv('TARGET_FPS',15))
print(TARGET_FPS)   

CAMERA_STREAMS = [
            {"id": 2272, "url": "PvzHLoVYSbD6qlFK186086798029876"},
            {"id": 2273, "url": "uJdAMnOwlctRrMFx422209601650"},
            {"id": 2274, "url": "PvzHLoVYSbD6qlFK186086798029876"}
            # {"id": 2270, "url": "rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/102"},
            # {"id": 1, "url": "rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/402"},
            # {"id": 2, "url": "rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/402"},
            # {"id": 3, "url": "rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/102"},
            # {"id": 4, "url": "rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/102"},
            # {"id": 5, "url": "rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/102"},
            # {"id": 6, "url": "rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/102"},
            # {"id": 7, "url": "rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/102"},
            # {"id": 8, "url": "rtsp://user2:admin1234@96.70.93.78:554/Streaming/channels/102"}
        ]