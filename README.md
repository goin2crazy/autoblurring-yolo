# YOLO Video and Image Processing  

## Features  
- **Blur Detected Zones**: Automatically blur detected objects in videos and images.  
- **Configurable Parameters**: Customize paths, confidence thresholds, and weights through command-line arguments.  
- **Efficient Video Processing**: Handles video frame-by-frame and outputs the processed video in `.mp4` format.  

---

## Requirements  
Install the necessary Python dependencies:  
```bash  
pip install ultralytics opencv-python numpy  
```  

---

## Usage  

### 1. Running the Script  
Run the script from the terminal using the following format:  
```bash  
python main.py --video_path <path_to_video> --image_path <path_to_image> --weights_path <path_to_weights> --conf <confidence_threshold>  
```  

### Arguments  
| Argument        | Description                                   | Required | Default      |  
|------------------|-----------------------------------------------|----------|--------------|  
| `--video_path`   | Path to the input video file                 | Yes      | N/A          |  
| `--image_path`   | Path to the input image file                 | Yes      | N/A          |  
| `--weights_path` | Path to the YOLOv8 model weights file        | Yes      | N/A          |  
| `--conf`         | Confidence threshold for object detection    | No       | `0.2`        |  

---

### 2. Example  
To process a video with the `best.pt` YOLO weights and a confidence threshold of 0.25, run:  
```bash  
python main.py --video_path ./example_video.mp4 --image_path ./example_image.jpg --weights_path ./best.pt --conf 0.25  
```  

---

## Output  
The script generates the following outputs:  
1. **Blurred Video**: The processed video with blurred zones is saved in the current directory as `output_video.mp4`.  
2. **Blurred Image**: The processed image with blurred zones is saved as `output_image.jpg`.  

---

## Code Structure  
The project is designed to ensure ease of modification and scalability:  
- **`main.py`**: Entry point for the script, using argparse for configurable arguments.  
- **`video_processor.py`**: Contains logic for processing videos frame by frame.  
- **`image_processor.py`**: Handles image processing and blurring of detected zones.  
- **`utils.py`**: Helper functions like `blur_zones` for zone blurring.  

---

## Notes  
- The model must be trained and saved as a `.pt` file (e.g., `best.pt`).  
- Ensure your video and image paths exist and are accessible by the script.  
- Tweak the `--conf` parameter to optimize detection sensitivity based on your needs.  

---

## Future Enhancements  
- Add support for real-time video processing via webcam.  
- Implement multi-threading for faster video processing.  
- Extend functionality to save metadata (e.g., detected object counts).  
