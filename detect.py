import cv2
import numpy as np
import threading
import time
import os
import ai_edge_litert.interpreter as litert
from picamera2 import Picamera2

# --- THREADED CAMERA CAPTURE VIA PICAMERA2 ---
class HeadlessVideoStream:
    def __init__(self, resolution=(320, 240)):
        print("[INFO] Initializing Picamera2 backend...")
        self.picam2 = Picamera2()
        
        config = self.picam2.create_video_configuration(
            main={'size': resolution, 'format': 'BGR888'}
        )
        self.picam2.configure(config)
        
        self.frame = None
        self.grabbed = False
        self.stopped = False

    def start(self):
        self.picam2.start()
        # Warm capture
        self.frame = self.picam2.capture_array()
        self.grabbed = True
        
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                # Capture frame as a clean NumPy array optimized for OpenCV / LiteRT
                rgb_frame = self.picam2.capture_array()
                
                # Convert to BGR so your downstream drawing and imwrite logic remains identical
                self.frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                self.grabbed = True
            except Exception as e:
                print(f"[ERROR] Capture failed: {e}")
                self.stop()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.picam2.stop()

# --- INITIALIZATION ---
MODEL_PATH = "models/coco_ssd_mobile_v1/detect.tflite"
LABEL_PATH = "models/coco_ssd_mobile_v1/labelmap.txt"
CONF_THRESHOLD = 0.5

# Load label map safely
with open(LABEL_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]
    # Remove metadata lines if any exist (like '???')
    if CLASSES and CLASSES[0] == '???':
        CLASSES.pop(0)

# Load LiteRT Model & allocate buffers using the new ai_edge_litert namespace
print("[INFO] Initializing LiteRT Engine...")
interpreter = litert.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get model tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
input_dtype = input_details[0]['dtype']

print("[INFO] Starting headless camera feed...")
vs = HeadlessVideoStream(resolution=(320, 240)).start()
time.sleep(2.0)  # Camera warm-up

last_save_time = time.time()
frame_count = 0
fps_start_time = time.time()

print("[INFO] Cat-cam LiteRT tracker running. Press Ctrl+C to exit.")

try:
    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame_count += 1
        current_time = time.time()
        should_save_image = (current_time - last_save_time) >= 10.0

        (h, w) = frame.shape[:2]

        # Prepare input matrix structure for LiteRT
        input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        input_resized = cv2.resize(input_rgb, (input_width, input_height))
        input_data = np.expand_dims(input_resized, axis=0)

        # Explicitly handle normalization if the model expects FLOAT32 instead of UINT8 quant
        if input_dtype == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        # Set input tensor and run execution
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Extract output tensor metrics (Fixed indices for standard SSD models)
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        terminal_output = []

        for i in range(len(scores)):
            confidence = float(scores[i])

            if confidence > CONF_THRESHOLD:
                idx = int(classes[i])
                label_text = CLASSES[idx] if idx < len(CLASSES) else f"Unknown ({idx})"

                # LiteRT SSD outputs normalized coords as: [ymin, xmin, ymax, xmax]
                startY = int(max(1, boxes[i][0] * h))
                startX = int(max(1, boxes[i][1] * w))
                endY = int(min(h, boxes[i][2] * h))
                endX = int(min(w, boxes[i][3] * w))

                box_str = f"[{label_text} ({confidence*100:.1f}%): Box({startX},{startY} -> {endX},{endY})]"
                terminal_output.append(box_str)

                if should_save_image:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, f"{label_text}: {confidence*100:.1f}%", (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if terminal_output:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Detections: " + ", ".join(terminal_output))

        if should_save_image:
            filename = "debug_output.jpg"
            cv2.imwrite(filename, frame)
            print(f" >>> [DEBUG] Saved snapshot to {os.path.abspath(filename)}")
            last_save_time = current_time

        if (current_time - fps_start_time) >= 5.0:
            calculated_fps = frame_count / (current_time - fps_start_time)
            print(f" --- [Performance] LiteRT Loop running at: {calculated_fps:.2f} FPS ---")
            frame_count = 0
            fps_start_time = current_time

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[INFO] Exiting program...")

finally:
    vs.stop()

