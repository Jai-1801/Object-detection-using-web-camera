# Object Detection Using Webcam 📸

## AIM
To write a Python code to perform object detection using a webcam.

---
## PROCEDURE

1.  **Load Network:** Load the pre-trained YOLOv4 network (weights and configuration files) using `cv2.dnn.readNet()`.
2.  **Load Class Labels:** Read the class labels from the `coco.names` file, which correspond to the objects the YOLOv4 model can detect.
3.  **Get Output Layers:** Identify the output layers of the YOLO network. These are needed to get the detection results after a forward pass.
4.  **Initialize Webcam:** Start video capture from the default webcam using `cv2.VideoCapture(0)`.
5.  **Process Frames:** For each frame captured from the webcam:
    * Convert the frame into a format suitable for YOLO (a "blob") using `cv2.dnn.blobFromImage()`.
    * Set this blob as the input to the network (`net.setInput()`).
    * Perform a forward pass through the network to get the raw detection data (`net.forward()`).
    * Parse this raw data to extract bounding box coordinates, confidence scores, and class IDs for all detected objects.
6.  **Apply Non-Max Suppression (NMS):** Use NMS to filter out weak and overlapping bounding boxes, keeping only the most relevant ones.
7.  **Draw Detections:** For each valid detection, draw a bounding box around the object and label it with the class name and confidence score using `cv2.rectangle()` and `cv2.putText()`.
8.  **Display Output:** Show the processed frame (with detections) in an OpenCV window using `cv2.imshow()`.
9.  **Exit Condition:** Allow the user to quit the application by pressing the 'q' key.
10. **Cleanup:** After the loop finishes (or is exited), release the webcam resource (`cap.release()`) and close all OpenCV display windows (`cv2.destroyAllWindows()`).

---
## PROGRAM

```python
# --- 1. Import necessary libraries ---
import os
import cv2  # OpenCV for computer vision tasks
import numpy as np  # NumPy for numerical operations
import urllib.request  # For downloading files

# --- 2. Define paths for YOLOv4 files ---
cfg_url = "[https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg)"
weights_url = "[https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)"
names_url = "[https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names)"

# Local paths (files will be in the same directory as the script)
cfg_path = "yolov4.cfg"
weights_path = "yolov4.weights"
names_path = "coco.names"

# --- 3. Download YOLOv4 configuration and weights files if they don't exist ---
def download_file_if_not_exists(url, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading {url} to {file_path}...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"Successfully downloaded {file_path}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            # Critical file, so raise error or exit if download fails
            raise SystemExit(f"Failed to download {file_path}. Exiting.")
    else:
        print(f"{file_path} already exists. Skipping download.")

print("Checking and downloading YOLOv4 files if necessary...")
download_file_if_not_exists(cfg_url, cfg_path)
download_file_if_not_exists(weights_url, weights_path)
download_file_if_not_exists(names_url, names_path)

# --- 4. Load YOLOv4 network ---
print("Loading YOLOv4 network...")
net = cv2.dnn.readNet(weights_path, cfg_path)
if net.empty():
    print("Failed to load YOLOv4 network. Check file paths and integrity.")
    raise SystemExit("YOLOv4 network loading failed. Exiting.")
print("YOLOv4 network loaded.")

# --- 5. Load COCO class labels ---
print("Loading COCO class labels...")
try:
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"COCO names file not found at {names_path}")
    raise SystemExit("COCO names file missing. Exiting.")
print(f"COCO class labels loaded ({len(classes)} classes).")

# --- 6. Get output layer names ---
layer_names = net.getLayerNames()
try:
    # Attempt for newer OpenCV versions (net.getUnconnectedOutLayers() returns 1D array)
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    # Fallback for older OpenCV versions (net.getUnconnectedOutLayers() returns 2D array)
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("Output layers identified.")

# --- 7. Initialize Webcam ---
print("Initializing webcam...")
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit("Webcam initialization failed. Exiting.")
print("Webcam initialized.")

# --- 8. Main Loop for Object Detection ---
print("\n--- Starting Real-Time Object Detection (Press 'q' to quit) ---")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting loop.")
            break

        # Get image dimensions
        height, width, channels = frame.shape

        # Prepare the image for YOLOv4
        # Input size (416,416) is common for YOLOv4
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Get YOLO output
        outputs = net.forward(output_layers)

        # Initialize lists to store detected boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Process outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]  # Class scores start from 6th element
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold (e.g., 50%)
                    # Object detected
                    # YOLO returns center (x,y), width, and height relative to image size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left corner of the box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Max Suppression to eliminate redundant overlapping boxes
        # score_threshold (0.5) and nms_threshold (0.4) can be adjusted
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the image
        if len(indexes) > 0:
            # Ensure indexes is a flat array for consistent iteration
            if isinstance(indexes, np.ndarray) and indexes.ndim > 1:
                indexes = indexes.flatten()

            for i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence_val = confidences[i]
                color = (0, 255, 0)  # Green color for bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence_val:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the resulting frame
        cv2.imshow("Real-Time Object Detection", frame)

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF  # waitKey(1) means 1ms delay
        if key == ord('q'):
            print("Quitting...")
            break

except KeyboardInterrupt:
    print("\nDetection process interrupted by user (Ctrl+C).")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Release the webcam and destroy all OpenCV windows
    if 'cap' in locals() and cap.isOpened():  # Check if cap was initialized
        cap.release()
        print("Webcam released.")
    cv2.destroyAllWindows()
    print("OpenCV windows destroyed.")
    print("\nDetection process finished or stopped.")

```
## OUTPUT
![download](https://github.com/user-attachments/assets/1ad2dc24-6891-4c78-b0d5-43cc54f1b2c9)

## RESULT
Thus, the Python program to detect objects using a webcam has been successfully executed.
