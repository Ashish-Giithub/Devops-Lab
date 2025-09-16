import os
import cv2
import logging
import time
from dotenv import load_dotenv
# from app.stream.rabbitmq import RabbitMQ
# from app.stream.chunks_process import FrameProcessor
from antmedia1.webrtc_subscriber import AntMediaCamera
# from antmedia_old.antmedia_webrtc import AntMediaCamera
#
logger = logging.getLogger("Camera Initialize :: ")
 
class CameraInit:
    def __init__(self) -> None:
        load_dotenv()
        self.websocket_url = os.getenv("WEBSOCKET_URL", "ws://192.168.29.92:5080/WebRTCAppEE/websocket")
   
    def camera_init(self, client_type=None, rtsp_url=None, camera_id=None):
        """
        Initialize camera based on client type.
       
        Args:
            client_type (str): Type of client ("rabitmq", "chunks", "webrtc", "rtsp", etc.)
            rtsp_url (str): URL for the camera source (RTSP URL or stream ID for WebRTC)
            camera_id (str): Optional camera ID for identifying specific camera config
           
        Returns:
            Camera object of appropriate type
        """
        # if client_type == "rabitmq":
        #     logger.info(" ::: RABBITMQ INITIATED ::: ")
        #     video = RabbitMQ()
        #     video.connect()
           
        # elif client_type == "chunks":
        #     video = FrameProcessor()
           
        if client_type == "webrtc":
            # Default to the provided URL as stream_id
            stream_id = rtsp_url
               
            # If camera_id is provided, try to get camera-specific stream ID
            if camera_id:
                camera_specific_stream_id = os.getenv(f"ANTMEDIA_STREAM_ID_{camera_id}")
                if camera_specific_stream_id:
                    stream_id = camera_specific_stream_id
                    logger.info(f"Using camera-specific stream ID for camera {camera_id}: {stream_id}")
           
            # If no stream_id from parameters or camera-specific env var, try default
            if not stream_id:
                stream_id = os.getenv("ANTMEDIA_STREAM_ID")
               
            # Final check if we have a valid stream ID
            if not stream_id:
                logger.error(f"No stream ID found for WebRTC camera {camera_id}")
                return False
               
            # Get websocket URL (can be overridden for specific cameras if needed)
            websocket_url = os.getenv(f"WEBSOCKET_URL_{camera_id}", self.websocket_url) if camera_id else self.websocket_url
           
            if not websocket_url:
                logger.error(f"No WebSocket URL found for WebRTC camera {camera_id}")
                return False
               
            # Set buffer size (can be camera-specific)
            buffer_size = int(os.getenv(f"BUFFER_SIZE_{camera_id}", os.getenv("BUFFER_SIZE", "25")))
           
            logger.info(f"Initializing WebRTC camera {camera_id} with: websocket={websocket_url}, stream_id={stream_id}, buffer_size={buffer_size}")
           
            video = AntMediaCamera(
                websocket_url=websocket_url,
                stream_id=stream_id,
                buffer_size=buffer_size,
            )
 
            time.sleep(10)  # Wait for 1 second before starting the video stream
           
               
        else:
            # Default case - assume RTSP or other OpenCV-compatible source
            logger.info(f"Initializing camera {camera_id} with client_type={client_type}, URL={rtsp_url}")
            video = cv2.VideoCapture(rtsp_url)
               
        return video
   
    def print_values(self):
        logger.info("::: Checking environment variables :::")
        load_dotenv()
       
        # env_vars = dict(os.environ)
        # logger.info("Environment Variables:")
        # for key, value in env_vars.items():
        #     logger.info(f"{key}: {value}")
 