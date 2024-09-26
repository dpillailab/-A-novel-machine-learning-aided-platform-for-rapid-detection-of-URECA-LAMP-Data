# -*- coding: utf-8 -*-
"""
YOLOv8 Object Detection: Training, Testing, and Prediction Script

This script provides functionality to:
1. Train a YOLOv8 model on a custom LAMP dataset
2. Load a pre-trained YOLOv8 model (e.g., "best.pt")
3. Run predictions on a folder of test images
4. Visualize and save detection results

Created on Wed Sep 25 10:15:21 2024
@author: biniyamkahsay.mezgeb@ucalgary.ca
"""

import os
import cv2
import torch
import argparse
from tqdm import tqdm
from ultralytics import YOLO

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_yolo_model(data_path, epochs=130, img_size=640, batch_size=16):
    """
    Train a YOLOv8 model on a custom LAMP dataset.

    Args:
    data_path (str): Path to the dataset configuration file (data.yaml)
    epochs (int): Number of epochs to train for
    img_size (int): Input image size for the model
    batch_size (int): Batch size for training

    Returns:
    YOLO: Trained YOLO model
    """
    # Initialize the model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    
    # Set up training configuration
    config = {
        'data': data_path,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
    }
        
    # Train the model
    results = model.train(**config)
    
    # Evaluate the model
    val_results = model.val()
    
    print("Training completed. Validation results:", val_results)
    return model

def load_yolo_model(model_path):
    """
    Load a pre-trained YOLOv8 model.

    Args:
    model_path (str): Path to the pre-trained model file (e.g., "best.pt")

    Returns:
    YOLO: Loaded YOLO model
    """
    model = YOLO(model_path)
    print(f"Loaded pre-trained model from {model_path}")
    return model

def run_yolo_image_test_folder(model, test_folder, output_folder):
    """
    Run YOLOv8 predictions on a folder of test images and save visualized results.

    Args:
    model (YOLO): Trained or loaded YOLO model
    test_folder (str): Path to the folder containing test images
    output_folder (str): Path to save the output images with predictions
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
   
    color_palette = [
        (0, 0, 255),    # Red in BGR
        (255, 0, 0),    # Blue in BGR
    ]
    
    # Define visualization parameters
    box_thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.6
    font_thickness = 3
    
    # Iterate through each image in the test folder
    for image_name in tqdm(os.listdir(test_folder), desc="Processing test images"):
        # Skip non-image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Perform YOLOv8 model prediction on the current image
        image_path = os.path.join(test_folder, image_name)
        results = model(image_path)
        
        # Process and save the image with custom colors
        for result in results:
            # Get the original image
            img = cv2.imread(image_path)
            
            # Get bounding boxes, classes and confidence scores
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            # Draw bounding boxes with custom colors
            for box, cls, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box.astype(int)
                color = color_palette[int(cls) % len(color_palette)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)
                
                # Prepare label
                label = f"{result.names[int(cls)]} {score:.2f}"
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Draw filled rectangle for text background
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                
                # Add label
                cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
            
            # Save the processed image
            output_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_image_path, img)
    
    print(f"Processed images saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Training and Prediction Script")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Mode: 'train' for training, 'predict' for prediction")
    parser.add_argument("--data", type=str, help="Path to data.yaml file (required for training)")
    parser.add_argument("--model", type=str, help="Path to pre-trained model (required for prediction)")
    parser.add_argument("--test_folder", type=str, required=True, help="Path to test images folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save output images")
    
    args = parser.parse_args()

    if args.mode == "train":
        if not args.data:
            raise ValueError("--data argument is required for training mode")
        
        # Train the model
        trained_model = train_yolo_model(args.data)
        
        # Save the trained model
        trained_model.export(format="onnx")
        print("Model exported as ONNX format")
        
        # Run predictions on test folder
        run_yolo_image_test_folder(trained_model, args.test_folder, args.output_folder)
    
    elif args.mode == "predict":
        if not args.model:
            raise ValueError("--model argument is required for prediction mode")
        
        # Load the pre-trained model
        loaded_model = load_yolo_model(args.model)
        
        # Run predictions on test folder
        run_yolo_image_test_folder(loaded_model, args.test_folder, args.output_folder)

    print("Script execution completed.")