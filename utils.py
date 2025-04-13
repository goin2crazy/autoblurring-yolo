import numpy as np 
import argparse
import cv2
from ultralytics import YOLO
from time import sleep
import os


from typing import List, Tuple, Optional

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

def extract_image_region(frame, zone):
    """Extracts a region from the frame based on the given zone.

    Args:
        frame (numpy.ndarray): The input frame.
        zone (list/tuple): [x_center, y_center, width, height] of the region.

    Returns:
        numpy.ndarray: The extracted region or None if the zone is invalid.
    """
    x_center, y_center, width, height = map(int, zone)
    x1 = max(0, x_center - width // 2)
    y1 = max(0, y_center - height // 2)
    x2 = min(frame.shape[1], x_center + width // 2)
    y2 = min(frame.shape[0], y_center + height // 2)

    if x1 < x2 and y1 < y2:
        return frame[y1:y2, x1:x2].copy()  # Return a copy to avoid modifying the original frame
    else:
        return None



def draw_rectangles(image: np.ndarray, zones: List[List[int]], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draws rectangles around specified zones in an image.

    Args:
        image (numpy.ndarray): The input image (cv2 image in BGR format).
        zones (list): A list of [x, y, w, h] coordinates representing the regions to draw rectangles around.
        color (tuple, optional): The color of the rectangles (B, G, R). Defaults to green (0, 255, 0).
        thickness (int, optional): The thickness of the rectangle borders. Defaults to 2.

    Returns:
        numpy.ndarray: The image with the rectangles drawn.
    """
    drawn_image = image.copy()
    for (x, y, w, h) in zones:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)


        
        x1 = max(0, int(x - w/2))
        y1 = max(0, int(y - h/2))

        x2 = min(image.shape[0]-1, (int(x + w/2)))
        y2 = min(image.shape[1]-1, int(y + h/2))

        cv2.rectangle(drawn_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return drawn_image


def images_to_video(image_list: List[np.ndarray], output_path: str, fps: float = 30.0) -> bool:
    """
    Converts a list of images (numpy.ndarrays) into a video file.  Handles images of different sizes.

    Args:
        image_list (list): A list of images (numpy.ndarrays).  All images should have the same color depth (e.g., RGB).
        output_path (str): The path to save the video file (e.g., "output.avi", "output.mp4").  The extension
                           determines the video format.
        fps (float, optional): Frames per second of the output video. Defaults to 30.0.

    Returns:
        bool: True if the video was successfully created, False otherwise.
    """
    if not image_list:
        print("Error: image_list is empty.  Cannot create video.")
        return False

    # Determine the maximum dimensions of the images.
    max_width = max(image.shape[1] for image in image_list)
    max_height = max(image.shape[0] for image in image_list)

    # Use the max dimensions for the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4, 'XVID' for .avi
    writer = cv2.VideoWriter(output_path, fourcc, fps, (max_width, max_height))  # Use max_width and max_height

    if writer is None:
        print(f"Error: Could not create VideoWriter for path: {output_path}")
        return False

    try:
        for image in image_list:
            # Resize each image to the maximum dimensions before writing to the video.  Use INTER_CONSTANT to pad with black.
            resized_image = cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_CONSTANT)
            writer.write(resized_image)
        writer.release()
        print(f"Video successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error writing video: {e}")
        writer.release()  # Ensure writer is released on error
        return False



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
