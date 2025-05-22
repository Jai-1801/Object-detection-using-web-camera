## Object-Detection-Using-WebCamera
## AIM:
To write a Python code to Object Detection Using Webcam.

## PROCEDURE:
STEP-1 Load the pre-trained YOLOv4 network (.weights and .cfg) using cv2.dnn.readNet().

STEP-2 Read class labels (COCO dataset) from the coco.names file.

STEP-3 Get the output layer names from the YOLO network using getLayerNames() and getUnconnectedOutLayers().

STEP-4 Start webcam video capture using cv2.VideoCapture(0).

STEP-5 Process each frame:
Convert the frame to a YOLO-compatible input using cv2.dnn.blobFromImage(). Pass the blob into the network (net.setInput()) and run forward pass to get detections (net.forward()). Parse the output to extract bounding boxes, confidence scores, and class IDs for detected objects. 

STEP-6 Use NMS to remove overlapping bounding boxes and retain the best ones.

STEP-7 Draw bounding boxes and labels on detected objects using cv2.rectangle() and cv2.putText().

STEP-8 Show the processed video frames with object detections using cv2.imshow().

STEP-9 Exit the loop if the 'q' key is pressed.

STEP-10 Release the video capture and close any OpenCV windows (cap.release() and cv2.destroyAllWindows()).
