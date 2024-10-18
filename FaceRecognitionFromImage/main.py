import cv2
import numpy as np

# Load the pre-trained deep learning model
modelFile = "res_ssd_300Dim.caffeModel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load the image (change the path to your image)
imagePath = "image.jpg"  # Replace with your image path
image = cv2.imread(imagePath)

if image is None:
    print("Error: Could not load image.")
    exit()

# Get image height and width
(h, w) = image.shape[:2]

# Prepare the image for the deep learning model
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False)

# Set the blob as input to the network
net.setInput(blob)

# Forward pass to get the detections
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
    # Extract the confidence (probability) associated with the detection
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections
    if confidence > 0.5:
        # Compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bounding box around the face
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 5)

        # Optionally display the confidence
        text = f"{confidence * 100:.2f}%"
        cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# Save the output image
outputPath = "output_image.jpg"  # The output file
cv2.imwrite(outputPath, image)

# Display the image with detections
cv2.imshow("DNN Face Detection - Image", image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
