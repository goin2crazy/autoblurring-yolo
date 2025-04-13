import threading
import queue
import numpy as np 
import argparse
import cv2
from ultralytics import YOLO
from time import sleep
import os

from utils import * 
from processing import * 

def handle_online_video(model, conf, display_type):
    """Handles online video processing from the laptop camera with different display options.

    Args:
        model (YOLO): The loaded YOLO model.
        conf (float): Confidence threshold for the YOLO model.
        display_type (str): The type of display ("all", "blur", "rectangle", "detect_area").
    """
    cap = cv2.VideoCapture(0)  # 0 for default camera
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Create windows, use namedWindow instead of imshow for more control
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original', 640, 480)  # You can adjust the size as needed

    if display_type == "all":
        cv2.namedWindow('Blurred', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Blurred', 640, 480)
        cv2.namedWindow('Rectangles', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Rectangles', 640, 480)
        cv2.namedWindow('Detected Area', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detected Area', 640, 480)


    output_frames = queue.Queue() # Use a queue to store the processed frames.

    def processing_thread_function():
        while True:
            ret, frame = cap.read()
            if not ret:
                output_frames.put(None) # Signal end of stream
                break

            processed_frame, zones = process_frame(frame, model, conf)
            output_frames.put((frame, processed_frame, zones)) # Put a tuple

    processing_thread = threading.Thread(target=processing_thread_function)
    processing_thread.daemon = True # Set as daemon so it exits when the main thread exits.
    processing_thread.start()


    while True:
        frames = output_frames.get()
        if frames is None:
            break # Exit loop

        original_frame, processed_frame, zones = frames

        if display_type == "all":
            blurred_frame = blur_zones(original_frame, zones) # use original frame here
            rect_frame = draw_rectangles(original_frame, zones) # use original frame here
            if zones:
                detected_area = extract_image_region(original_frame, zones[0]) # Extract only the first detected zone.
            else:
                detected_area = np.zeros_like(original_frame) # black frame if no detection


            cv2.imshow('Original', original_frame)
            cv2.imshow('Blurred', blurred_frame)
            cv2.imshow('Rectangles', rect_frame)
            if zones: # only show if there are detections
                cv2.imshow('Detected Area', detected_area)
        elif display_type == "blur":
            cv2.imshow('Original', original_frame)
            cv2.imshow('Blurred', processed_frame)
        elif display_type == "rectangle":
            rect_frame = draw_rectangles(original_frame, zones)
            cv2.imshow('Original', original_frame)
            cv2.imshow('Rectangles', rect_frame)
        elif display_type == "detect_area":
            cv2.imshow('Original', original_frame)
            if zones:
                cv2.imshow('Detected Area', extract_image_region(original_frame, zones[0]))
            else:
                #show a black image
                cv2.imshow('Detected Area', np.zeros_like(original_frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    processing_thread.join() # wait for the thread to finish
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO-based blurring on images or videos.")
    parser.add_argument("--video_path", type=str, required=False, default='none', help="Path to the input video file.")
    parser.add_argument("--image_path", type=str, required=False, default='none', help="Path to the input image file.")
    parser.add_argument("--weights_path", type=str, required=False, default='best.pt', help="Path to the YOLO weights file.")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold for YOLO model.")
    parser.add_argument("--online", action="store_true", help="Enable online video processing from camera.")
    parser.add_argument("--type", type=str, default="blur", choices=["all", "blur", "rectangle", "detect_area"],
                        help="Type of display for online processing: all, blur, rectangle, detect_area.")
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

    # Handle online video if enabled
    if args.online:
        handle_online_video(model, args.conf, args.type)
