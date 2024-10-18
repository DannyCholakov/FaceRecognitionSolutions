import cv2
import numpy as np

# Load YOLO
modelFile = "yolov3.weights"
configFile = "yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configFile, modelFile)

# Load the class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the video file
video_file = "video.mp4"  # Change this to the path of your video file
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the names of the output layers
layer_names = net.getLayerNames()
# Check for the version of OpenCV
out_layer_indices = net.getUnconnectedOutLayers()
if isinstance(out_layer_indices, np.ndarray):
    output_layers = [layer_names[i - 1] for i in out_layer_indices.flatten()]  # Handle numpy array
else:
    output_layers = [layer_names[i - 1] for i in out_layer_indices]  # Handle list

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))  # Resize to 640x480 or any size you prefer

    # Prepare the image for the deep learning model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Run forward pass to get the detections
    detections = net.forward(output_layers)

    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections (confidence > threshold)
            if confidence > 0.7 and class_id == 0:  # 0 is for 'person'
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

                # Display the confidence
                text = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Display the output frame
    cv2.imshow("YOLO Face Detection on Video", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
