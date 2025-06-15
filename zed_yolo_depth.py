import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import math

# === Load YOLOv8 model ===

# Make sure to update this line in the script with your YOLOv8 model path
model = YOLO("")

# === Initialize ZED camera ===
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_resolution = sl.RESOLUTION.HD720

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("ZED camera not detected!")
    exit(1)

runtime_params = sl.RuntimeParameters()
image_zed = sl.Mat()
depth = sl.Mat()

# === Main Loop ===
print("Press 'q' to quit.")
while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # Get image and depth data
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.XYZ)

        frame = image_zed.get_data()
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Resize for YOLO inference
        resized_frame = cv2.resize(frame, (640, 360))
        results = model(resized_frame, verbose=False)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            idx = np.argmax(confs)
            x1, y1, x2, y2 = map(int, boxes.xyxy[idx].cpu().numpy())

            # Rescale box to original frame
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 360
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Get depth at center
            err, point_cloud_value = depth.get_value(cx, cy)
            depth_text = "Depth: {:.2f} m".format(point_cloud_value[2]) if math.isfinite(point_cloud_value[2]) else "Depth: N/A"

            # Draw bounding box and depth info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, depth_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show output
        cv2.imshow("YOLO + ZED", cv2.resize(frame, (960, 540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Clean up ===
zed.close()
cv2.destroyAllWindows()
