import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

class TRTEngine:
    def __init__(self, engine_path: str):
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers(self.engine)

    def _load_engine(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError("Failed to deserialize engine")
        return engine

    def _allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding_idx in range(engine.num_bindings):
            name = engine.get_binding_name(binding_idx)
            is_input = engine.binding_is_input(binding_idx)
            shape = tuple(engine.get_binding_shape(binding_idx))
            shape = tuple(s if s > 0 else 1 for s in shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(dev_mem))
            entry = {
                "name": name,
                "idx": binding_idx,
                "shape": shape,
                "dtype": dtype,
                "host": host_mem,
                "device": dev_mem,
            }
            if is_input:
                inputs.append(entry)
            else:
                outputs.append(entry)
        return inputs, outputs, bindings, stream

    def preprocess(self, image):
        _, c, h, w = self.inputs[0]["shape"]
        img_resized = cv2.resize(image, (w, h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        tensor = np.expand_dims(img_chw, axis=0)
        dtype = self.inputs[0]["dtype"]
        if dtype == np.float16:
            tensor = tensor.astype(np.float16)
        else:
            tensor = tensor.astype(np.float32)
        return np.ascontiguousarray(tensor)

    def infer(self, image):
        input_tensor = self.preprocess(image)
        host_input = self.inputs[0]["host"]
        np.copyto(host_input, input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], host_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        out_candidate = max(self.outputs, key=lambda o: int(np.prod(o["shape"])))
        return out_candidate["host"], self.inputs[0]["shape"], image.shape
    
    def decode(self, raw_output, input_shape, orig_shape):
        if len(input_shape) == 4:
            _, _, in_h, in_w = input_shape
        else:
            _, in_h, in_w = (1, input_shape[1], input_shape[2]) if len(input_shape) == 3 else (1, 640, 640)
        orig_h, orig_w = orig_shape[:2]
        boxes, scores, classes = self.decode_yolo_output(
            raw_output, input_w=in_w, input_h=in_h, orig_w=orig_w, orig_h=orig_h, conf_thr=0.25
        )
        return boxes, scores, classes
    
    def decode_yolo_output(self, raw_output: np.ndarray, input_w: int, input_h: int, orig_w: int, orig_h: int, conf_thr=0.25):
        print(f"[Decode] Decoding output tensor of size {raw_output.size}")
        arr = np.array(raw_output)
        total = arr.size

        possible = []
        for num_attrs in (84, 85, 85, 6):
            if total % num_attrs == 0:
                n = total // num_attrs
                possible.append((num_attrs, n))

        if not possible:
            print("[Decode] No suitable shape found, fallback to (N,6)")
            arr2 = arr.reshape(-1, 6)
            boxes, scores, classes = [], [], []
            for row in arr2:
                x, y, w, h, conf, cls = row
                if conf < conf_thr:
                    continue
                if max(x, y, w, h) <= 1.01:
                    cx = x * input_w
                    cy = y * input_h
                    bw = w * input_w
                    bh = h * input_h
                else:
                    cx, cy, bw, bh = x, y, w, h
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0
                x1 = x1 * (orig_w / input_w)
                x2 = x2 * (orig_w / input_w)
                y1 = y1 * (orig_h / input_h)
                y2 = y2 * (orig_h / input_h)
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
                classes.append(int(cls))
            if len(boxes) == 0:
                print("[Decode] No detections after confidence filtering")
                return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
            print(f"[Decode] Decoded {len(boxes)} boxes")
            return np.array(boxes), np.array(scores), np.array(classes, dtype=int)

        num_attrs, n = min(possible, key=lambda x: (x[0] != 84, x[0]))
        print(f"[Decode] Using shape ({num_attrs}, {n}) for decoding")

        if num_attrs in (84, 85):
            try:
                out = arr.reshape(1, num_attrs, n)
                out = out.transpose(0, 2, 1)[0]
            except Exception as e:
                print(f"[Decode] Exception during reshape-transpose: {e}, trying alternative reshape")
                out = arr.reshape(1, n, num_attrs)[0]
        else:
            out = arr.reshape(-1, num_attrs)

        boxes_xywh = out[:, :4]
        scores_all = out[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        class_scores = np.max(scores_all, axis=1)

        mask = class_scores > conf_thr
        if not mask.any():
            print("[Decode] No detections above confidence threshold")
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

        boxes_xywh = boxes_xywh[mask]
        class_scores = class_scores[mask]
        class_ids = class_ids[mask]

        boxes = []
        for (cx, cy, bw, bh) in boxes_xywh:
            if max(cx, cy, bw, bh) <= 1.01:
                cx = cx * input_w
                cy = cy * input_h
                bw = bw * input_w
                bh = bh * input_h
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
            x1 = x1 * (orig_w / input_w)
            x2 = x2 * (orig_w / input_w)
            y1 = y1 * (orig_h / input_h)
            y2 = y2 * (orig_h / input_h)
            boxes.append([x1, y1, x2, y2])

        print(f"[Decode] Decoded {len(boxes)} boxes after confidence filtering")
        return np.array(boxes, dtype=float), class_scores.astype(float), class_ids.astype(int)