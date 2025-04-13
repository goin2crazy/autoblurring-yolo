import threading
import queue
import numpy as np 
import argparse
import cv2
from ultralytics import YOLO
from time import sleep
import os

from utils import * 


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
        return blur_zones(frame, zones), zones # Return both the blurred frame and the zones
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, [] # Return the original frame and an empty list of zones in case of an error



def process_video(video_path, output_path, model, conf, fps=30, display_type='none'):
    """
    Process a video, frame by frame, using the YOLO model and save the processed video.
    Optionally display the processed video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the processed video.
        model (YOLO): The loaded YOLO model.
        conf (float): Confidence threshold for the YOLO model.
        fps (int): Frames per second for the output video.
        display_type (str):  "none", "all", "blur", "rectangle", "detect_area".
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video at {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width // 2, frame_height // 2)) # Divide frame size by 2

    # Create windows if needed
    if display_type in ["all", "blur", "rectangle", "detect_area"]:
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original', 640, 480)
    if display_type == "all":
        cv2.namedWindow('Combined', cv2.WINDOW_NORMAL) # Changed window name
        cv2.resizeWindow('Combined', 640, 480)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video finished.")
            break
        processed_frame, zones = process_frame(frame, model, conf) # Get zones

        if display_type == 'none':
            out.write(processed_frame)
        else:
            if display_type == "all":
                blurred_frame = blur_zones(frame, zones)
                rect_frame = draw_rectangles(frame, zones)
                if zones:
                    detected_area = extract_image_region(frame, zones[0])
                else:
                    detected_area = np.zeros_like(frame)

                # Resize frames to half their original size
                frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
                blurred_frame = cv2.resize(blurred_frame, (frame_width // 2, frame_height // 2))
                rect_frame = cv2.resize(rect_frame, (frame_width // 2, frame_height // 2))
                detected_area = cv2.resize(detected_area, (frame_width // 2, frame_height // 2))

                # Create a combined frame
                combined_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                combined_frame[:frame_height // 2, :frame_width // 2] = frame
                combined_frame[:frame_height // 2, frame_width // 2:] = blurred_frame
                combined_frame[frame_height // 2:, :frame_width // 2] = rect_frame
                combined_frame[frame_height // 2:, frame_width // 2:] = detected_area

                cv2.imshow('Combined', combined_frame) # Show combined frame
                out.write(combined_frame) # Save combined frame
            elif display_type == "blur":
                cv2.imshow('Original', frame)
                cv2.imshow('Blurred', processed_frame)
                out.write(processed_frame)
            elif display_type == "rectangle":
                rect_frame = draw_rectangles(frame, zones)
                cv2.imshow('Original', frame)
                cv2.imshow('Rectangles', rect_frame)
                out.write(processed_frame)
            elif display_type == "detect_area":
                cv2.imshow('Original', frame)
                if zones:
                    cv2.imshow('Detected Area', extract_image_region(frame, zones[0]))
                else:
                    cv2.imshow('Detected Area', np.zeros_like(frame))
                out.write(processed_frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
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

