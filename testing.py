# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:22:15 2024
Modified for high-resolution publication quality

@author: biniyamkahsay.mezgeb
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from tqdm import tqdm

def run_yolo_image_test_folder(model_prediction, test_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define custom color palette with high contrast colors
    color_palette = [
        (0, 0, 255),    # Red in BGR
        (0, 255, 0),    # Green in BGR
        (255, 0, 0),    # Blue in BGR
        (0, 255, 255),  # Yellow in BGR
        (255, 0, 255),  # Magenta in BGR
        (255, 255, 0),  # Cyan in BGR
    ]
    
 
    box_thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2  # Adjusted for better readability
    font_thickness = 2
    
    # Label styling parameters
    text_padding = 8  # Padding around text
    outline_thickness = 4  # Thickness of text outline

    # Iterate through each image in the test folder
    for image_name in tqdm(os.listdir(test_folder)):
        # Skip non-image files
        if not image_name.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Perform YOLOv8 model prediction on the current image
        image_path = os.path.join(test_folder, image_name)
        results = model_prediction(image_path)

        # Process and save the image with enhanced labels
        for result in results:
            # Get the original image
            img = cv2.imread(image_path)
            
            # Get bounding boxes, classes and confidence scores
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            # Draw bounding boxes with enhanced labels
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box.astype(int)
                color = color_palette[int(cls) % len(color_palette)]
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

                # Prepare label
                label = f"{result.names[int(cls)]} {score:.2f}"
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Calculate label background rectangle
                bg_x1 = x1
                bg_y1 = y1 - text_height - text_padding * 2
                bg_x2 = x1 + text_width + text_padding * 2
                bg_y2 = y1
                
                # Ensure background doesn't go outside image bounds
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(img.shape[1], bg_x2)
                
                # Create a solid opaque background for maximum readability
                # Use white background with black text for best contrast
                cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)  # White background
                cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 3)  # Colored border
                
                # Add text with high contrast
                text_x = x1 + text_padding
                text_y = y1 - text_padding
                
                # Draw black text on white background for maximum readability
                cv2.putText(img, label, (text_x, text_y), font, font_scale, 
                           (0, 0, 0), font_thickness)

            # Save the processed image with high quality
            output_image_path = os.path.join(output_folder, image_name)
            # Use high quality JPEG settings
            cv2.imwrite(output_image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

def run_yolo_high_contrast_labels(model_prediction, test_folder, output_folder):
    """
    Version optimized for maximum label visibility on any background
    """
    os.makedirs(output_folder, exist_ok=True)

    color_palette = [
        (0, 0, 255),    # Red in BGR
        (0, 255, 0),    # Green in BGR
    ]
    
    # Larger, bolder settings for better visibility
    box_thickness = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4  # Larger font
    font_thickness = 3  # Thicker font
    text_padding = 12  # More padding

    for image_name in tqdm(os.listdir(test_folder)):
        if not image_name.endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(test_folder, image_name)
        results = model_prediction(image_path)

        for result in results:
            img = cv2.imread(image_path)
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box.astype(int)
                color = color_palette[int(cls) % len(color_palette)]
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

                # Prepare label
                label = f"{result.names[int(cls)]} {score:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Calculate background dimensions with extra padding
                bg_x1 = x1 - 2
                bg_y1 = y1 - text_height - text_padding * 2 - 5
                bg_x2 = x1 + text_width + text_padding * 2 + 5
                bg_y2 = y1 + 2
                
                # Ensure background stays within image bounds
                bg_x1 = max(0, bg_x1)
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(img.shape[1], bg_x2)
                
                # Create solid white background with thick colored border
                cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)  # White fill
                cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 4)  # Thick colored border
                cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), 2)  # Thin black inner border
                
                # Add text with maximum contrast
                text_x = x1 + text_padding
                text_y = y1 - text_padding - 3
                
                # Black text on white background for ultimate readability
                cv2.putText(img, label, (text_x, text_y), font, font_scale, 
                           (0, 0, 0), font_thickness)

            # Save with high quality
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])



    """
    Alternative version with different label styling approaches
    """
    os.makedirs(output_folder, exist_ok=True)

    color_palette = [
        (0, 0, 255),    # Red in BGR
        (0, 255, 0),    # Green in BGR
    ]
    
    box_thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2

    for image_name in tqdm(os.listdir(test_folder)):
        if not image_name.endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(test_folder, image_name)
        results = model_prediction(image_path)

        for result in results:
            img = cv2.imread(image_path)
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box.astype(int)
                color = color_palette[int(cls) % len(color_palette)]
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

                # Prepare label
                label = f"{result.names[int(cls)]} {score:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Style 1: Solid background with contrasting text
                bg_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                text_color = (0, 0, 0) if bg_color == (255, 255, 255) else (255, 255, 255)
                
                # Draw solid background
                cv2.rectangle(img, (x1, y1 - text_height - 12), 
                            (x1 + text_width + 12, y1), bg_color, -1)
                
                # Draw border around background
                cv2.rectangle(img, (x1, y1 - text_height - 12), 
                            (x1 + text_width + 12, y1), color, 2)
                
                # Add text
                cv2.putText(img, label, (x1 + 6, y1 - 6), font, font_scale, 
                           text_color, font_thickness)

            # Save with high quality
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

# Example usage
if __name__ == "__main__":
    model_prediction = YOLO("best.pt")
    test_folder = r"D:\yolo Ryan\PlosOne_publication_codes\Isolates\Isolates"
    output_folder = r"D:\yolo Ryan\PlosOne_publication_codes\Isolates\Isolates_detection"
    
    # Use the main function with enhanced labels
    run_yolo_image_test_folder(model_prediction, test_folder, output_folder)
    
    # Or try the alternative styling
    # run_yolo_image_test_folder_alternative(model_prediction, test_folder, output_folder)