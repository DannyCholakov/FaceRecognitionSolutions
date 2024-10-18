# Face Detection Application using OpenCV and YOLO

## Overview
This project implements a real-time face detection application utilizing OpenCV and the YOLO (You Only Look Once) deep learning model. The application can process video streams from a webcam or read from video files, detecting and outlining faces with colored bounding boxes. The application aims to demonstrate the capabilities of computer vision and deep learning for real-time applications.

## Features
- **Real-time Face Detection**: Detects faces in live webcam feeds or video files.
- **Bounding Boxes**: Draws colored rectangles around detected faces, providing immediate visual feedback.
- **Configurable**: Supports various video sources, including local video files and webcam inputs.
- **Performance Optimizations**: Includes frame skipping and resizing to improve processing speed without significantly compromising accuracy.

## Requirements
- Python 3.x
- OpenCV
- Numpy
- YOLO model weights and configuration files (YOLOv3 or YOLOv4-tiny recommended)
- COCO class labels (`coco.names`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FaceDetectionApp.git
   cd FaceDetectionApp
   ```
2. Install the required packages:
   ```bash
   pip install opencv-python numpy
   ```
3. Download the YOLO model weights and configuration files:
   - [YOLOv3.weights](https://pjreddie.com/media/files/yolov3.weights)
   - [YOLOv3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

4. Place the downloaded files in the project directory.

## Usage
Run the application with the following command:
```bash
python main.py
```
- For webcam detection, ensure the webcam is connected and functional.
- For video file detection, update the `video_file` variable in `main.py` with the path to your video file.

## Contributions
Feel free to submit issues or pull requests for improvements, bug fixes, or enhancements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [OpenCV](https://opencv.org/) for providing the computer vision library.
- [YOLO](https://pjreddie.com/darknet/yolo/) for the state-of-the-art object detection algorithm.
