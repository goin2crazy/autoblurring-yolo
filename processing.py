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
    print(f"Display type: {display_type}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video at {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_base = os.path.splitext(output_path)[0]
    writers = {}
    output_paths = []

    # Initialize VideoWriters based on display_type
    if display_type != 'none':
        output_original = f"{output_base}_original.mp4"
        writers['original'] = cv2.VideoWriter(output_original, fourcc, fps, (frame_width, frame_height))
        output_paths.append(output_original)

    if display_type in ['all', 'blur']:
        output_blurred = f"{output_base}_blurred.mp4"
        writers['blurred'] = cv2.VideoWriter(output_blurred, fourcc, fps, (frame_width, frame_height))
        print(output_paths)
        output_paths.append(output_blurred)

    if display_type in ['all', 'rectangle']:
        output_rectangles = f"{output_base}_rectangles.mp4"
        writers['rectangles'] = cv2.VideoWriter(output_rectangles, fourcc, fps, (frame_width, frame_height))
        print(output_paths)
        output_paths.append(output_rectangles)

    if display_type in ['all', 'detect_area']:
        output_detected_area = f"{output_base}_detected_area.mp4"
        writers['detected_area'] = cv2.VideoWriter(output_detected_area, fourcc, fps, (frame_width, frame_height))
        print(output_paths)
        output_paths.append(output_detected_area)

    # Create windows for display
    windows = []
    if display_type != 'none':
        windows.append('Original')
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original', 640, 480)
        
    if display_type == 'all':
        windows.extend(['Blurred', 'Rectangles', 'Detected Area'])
        for win in ['Blurred', 'Rectangles', 'Detected Area']:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 640, 480)
    elif display_type == 'blur':
        windows.append('Blurred')
        cv2.namedWindow('Blurred', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Blurred', 640, 480)
    elif display_type == 'rectangle':
        windows.append('Rectangles')
        cv2.namedWindow('Rectangles', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Rectangles', 640, 480)
    elif display_type == 'detect_area':
        windows.append('Detected Area')
        cv2.namedWindow('Detected Area', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detected Area', 640, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame to get detection zones
        _, zones = process_frame(frame, model, conf)

        # Always write original frame if needed
        if 'original' in writers:
            writers['original'].write(frame)

        # Handle different display types
        if display_type == 'all':
            blurred = blur_zones(frame.copy(), zones)
            rectangles = draw_rectangles(frame.copy(), zones)
            detected = extract_image_region(frame, zones[0]) if zones else np.zeros_like(frame)
            
            writers['blurred'].write(blurred)
            writers['rectangles'].write(rectangles)
            writers['detected_area'].write(detected)
            
            cv2.imshow('Original', frame)
            cv2.imshow('Blurred', blurred)
            cv2.imshow('Rectangles', rectangles)
            cv2.imshow('Detected Area', detected)

        elif display_type == 'blur':
            blurred = blur_zones(frame.copy(), zones)
            writers['blurred'].write(blurred)
            cv2.imshow('Original', frame)
            cv2.imshow('Blurred', blurred)

        elif display_type == 'rectangle':
            rectangles = draw_rectangles(frame.copy(), zones)
            writers['rectangles'].write(rectangles)
            cv2.imshow('Original', frame)
            cv2.imshow('Rectangles', rectangles)

        elif display_type == 'detect_area':
            detected = extract_image_region(frame, zones[0]) if zones else np.zeros_like(frame)
            writers['detected_area'].write(detected)
            cv2.imshow('Original', frame)
            cv2.imshow('Detected Area', detected)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Cleanup
    cap.release()
    for writer in writers.values():
        writer.release()
    # cv2.destroyAllWindows()

    print(f"Processed videos saved to: {', '.join(output_paths)}")


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

