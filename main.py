import argparse
import cv2
from ultralytics import YOLO
from time import sleep
import os

def load_model(weights_path):
    """
    Load the YOLO model from the given weights path.
    """
    try:
        return YOLO(weights_path)
    except Exception as e:
        raise ValueError(f"Error loading YOLO model: {e}")

def blur_zones(image, zones):
    """
    Blurs specific zones in an image.

    Args:
        image (numpy.ndarray): The input image (cv2 image in BGR format).
        zones (list): A list of [x, y, w, h] coordinates representing the regions to blur.

    Returns:
        numpy.ndarray: The image with the specified zones blurred.
    """
    blurred_image = image.copy()
    for (x, y, w, h) in zones:
        x = int(x)
        y = int(y)
        w = int(w * 1.2)
        h = int(h * 1.2)
        # Extract the region of interest (ROI)
        roi = blurred_image[max(0, y-int(h/2)):min(y+int(h/2), image.shape[0]),
                            max(0, x-int(w/2)):min(x+int(w/2), image.shape[1])]
        # Apply Gaussian blur to the ROI
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 30)
        # Replace the ROI in the image with the blurred ROI
        blurred_image[max(0, y-int(h/2)):min(y+int(h/2), image.shape[0]),
                      max(0, x-int(w/2)):min(x+int(w/2), image.shape[1])] = blurred_roi
    return blurred_image

def process_frame(frame, model, conf):
    """
    Process a single frame using the YOLO model and apply blurring to detected zones.

    Args:
        frame (numpy.ndarray): The video frame to process.
        model (YOLO): The loaded YOLO model.
        conf (float): Confidence threshold for the YOLO model.

    Returns:
        numpy.ndarray: The processed frame.
    """
    try:
        results = model.predict(frame, conf=conf, iou=1)
        zones = results[0].boxes.xywh.tolist()
        return blur_zones(frame, zones)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame

def process_video(video_path, output_path, model, conf, fps=30):
    """
    Process a video, frame by frame, using the YOLO model and save the processed video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the processed video.
        model (YOLO): The loaded YOLO model.
        conf (float): Confidence threshold for the YOLO model.
        fps (int): Frames per second for the output video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video at {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video finished.")
            break
        processed_frame = process_frame(frame, model, conf)
        out.write(processed_frame)

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")

def process_image(image_path, model, conf):
    """
    Process an image using the YOLO model and blur detected zones.

    Args:
        image_path (str): Path to the input image file.
        model (YOLO): The loaded YOLO model.
        conf (float): Confidence threshold for the YOLO model.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = cv2.imread(image_path)
    processed_image = process_frame(image, model, conf)
    output_path = os.path.splitext(image_path)[0] + "_processed.jpg"
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO-based blurring on images or videos.")
    parser.add_argument("--video_path", type=str, required=False, default='none', help="Path to the input video file.")
    parser.add_argument("--image_path", type=str, required=False, default='none',help="Path to the input image file.")
    parser.add_argument("--weights_path", type=str, required=False, default='best.pt', help="Path to the YOLO weights file.")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold for YOLO model.")
    args = parser.parse_args()

    # Load YOLO model
    model = load_model(args.weights_path)

    # Process video if provided
    if args.video_path != 'none':
        output_video_path = os.path.splitext(args.video_path)[0] + "_blurred.mp4"
        process_video(args.video_path, output_video_path, model, args.conf)

    # Process image if provided
    if args.image_path != 'none':
        process_image(args.image_path, model, args.conf)
