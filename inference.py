import os
import cv2
import logging
import numpy as np
import pycuda.autoinit
from datetime import datetime
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.point import Point


from trt_engine import TRTEngine



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("YOLOv8_TRT")

class TRT_YOLOv8:
    def __init__(
        self,
        engine_path: str,
        kafka_producer=None,
        region_points: list = None,
        camera_id: int = 0,
        conf_thresh: float = 0.25,
    ) -> None:
        self.max_outside_time = 30
        self.camera_id = camera_id
        self.conf_thresh = conf_thresh

        # Load TensorRT engine
        self.engine = TRTEngine(engine_path)
        logger.info("TRT_YOLOv8 initialized")


    def detect(self, image, verbose: bool = False) -> None:
        detections = []
        logger.debug("Detect method called")
        image = cv2.resize(image, (640, 480))
        raw_output, input_shape, orig_shape = self.engine.infer(image)
        boxes, scores, classes = self.engine.decode(raw_output, input_shape, orig_shape)
        # Only keep person class (usually class 0)
        person_mask = classes == 0
        boxes = boxes[person_mask]
        scores = scores[person_mask]
        classes = classes[person_mask]
        detections.append({
                            "class_id": classes,
                            "confidence": scores,
                            "bbox": boxes,
                        })
        return detections

        # # Tracking
        # track_ids = []
        # for bbox in boxes:
        #     # Convert to int for tracker
        #     bbox_int = [int(x) for x in bbox]
        #     track_id = self.tracker.update_id(self.camera_id, bbox_int, 0)
        #     track_ids.append(track_id)
        #     self.track_history[track_id].append(
        #         ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        #     )
        #     if len(self.track_history[track_id]) > 30:
        #         self.track_history[track_id].pop(0)

        # # Counting logic (same as before)
        # for track_id, track_line in self.track_history.items():
        #     if len(track_line) < 2:
        #         continue
        #     if self.counting_region.contains(Point(track_line[-1])):
        #         direction = track_line[-1][1] - np.mean([point[1] for point in track_line[:-1]])
        #         if direction > 0:
        #             detected_direction = "OUT" if self.reverse_direction else "IN"
        #         else:
        #             detected_direction = "IN" if self.reverse_direction else "OUT"
        #         if detected_direction not in self.counting_history[track_id]:
        #             self.counting_history[track_id].add(detected_direction)
        #             print("Detected direction:", detected_direction)
        #             # await self.produce_message(detected_direction)
        #     else:
        #         self.outside_region_timer[track_id] += 1
        #         if self.outside_region_timer[track_id] >= self.max_outside_time:
        #             self.counting_history.pop(track_id, None)
        #             self.outside_region_timer.pop(track_id, None)


    # async def produce_message(self, _type: str) -> None:
    #     message = PeopleCountingMessage(
    #         camera_id=os.getenv("CAMERA_ID", "001"),
    #         going_in=1 if _type == "IN" else 0,
    #         going_out=1 if _type == "OUT" else 0,
    #         timestamp=datetime.now(tz=tzinfo),
    #         model_version="v1.0.0"
    #     )
    #     logger.debug(f"Attempting to produce message: {message}")
    #     if self.kafka_producer:
    #         try:
    #             await self.kafka_producer.produce(
    #                 topic=os.getenv("KAFKA_TOPIC", "live-people-count"), message=message
    #             )
    #             logger.info(f"Message produced successfully: {message}")
    #         except Exception as e:
    #             logger.error(f"Failed to produce message: {e}")
    #     else:
    #         logger.warning("Kafka producer not initialized. Message not sent.")