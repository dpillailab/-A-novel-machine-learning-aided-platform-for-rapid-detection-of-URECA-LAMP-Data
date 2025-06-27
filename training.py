# -*- coding: utf-8 -*-
"""
Fixed YOLOv8 Setup for LAMP Dataset
"""

import os
import yaml
import torch
from ultralytics import YOLO

def create_correct_data_yaml():
    """
    Create the correct data.yaml file based on your actual folder structure
    """
    
    # Your base path (adjust if needed)
    base_path = r"D:\yolo Ryan\PlosOne_publication_codes\LAMP images"
    
    
    # Check if the base path exists
    if not os.path.exists(base_path):
        print(f"ERROR: Base path does not exist: {base_path}")
        return None
    
    # Check what's actually in your LAMP images folder
    print(f"Contents of {base_path}:")
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
            # Check if it has images and labels subfolders
            if os.path.exists(os.path.join(item_path, "images")):
                print(f"    📁 images/ ({len(os.listdir(os.path.join(item_path, 'images')))} files)")
            if os.path.exists(os.path.join(item_path, "labels")):
                print(f"    📁 labels/ ({len(os.listdir(os.path.join(item_path, 'labels')))} files)")
        else:
            print(f"  📄 {item}")
    
    # Check if training and validation folders exist and have the right structure
    train_path = os.path.join(base_path, "training")
    val_path = os.path.join(base_path, "validation")
    
    if not os.path.exists(train_path):
        print(f"ERROR: Training folder not found: {train_path}")
        return None
        
    if not os.path.exists(val_path):
        print(f"ERROR: Validation folder not found: {val_path}")
        return None
    
    # Create data.yaml with absolute paths to avoid path issues
    data_config = {
        'path': base_path.replace('\\', '/'),  # Convert to forward slashes
        'train': 'training',
        'val': 'validation',
        'nc': 2,
        'names': ['positive', 'negative']
    }
    
    yaml_file = 'data.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"\n✓ Created corrected data.yaml:")
    with open(yaml_file, 'r') as f:
        print(f.read())
    
    return yaml_file

def check_dataset_structure(base_path):
    """
    Check if your dataset has the correct YOLO structure
    """
    print("\n" + "="*50)
    print("CHECKING DATASET STRUCTURE")
    print("="*50)
    
    required_structure = {
        'training': ['images', 'labels'],
        'validation': ['images', 'labels']
    }
    
    all_good = True
    
    for split, subfolders in required_structure.items():
        split_path = os.path.join(base_path, split)
        print(f"\nChecking {split} folder:")
        
        if not os.path.exists(split_path):
            print(f"  ❌ {split} folder missing!")
            all_good = False
            continue
            
        for subfolder in subfolders:
            subfolder_path = os.path.join(split_path, subfolder)
            if os.path.exists(subfolder_path):
                file_count = len([f for f in os.listdir(subfolder_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.txt'))])
                print(f"  ✓ {subfolder}/ ({file_count} files)")
            else:
                print(f"  ❌ {subfolder}/ folder missing!")
                all_good = False
    
    if all_good:
        print("\n✅ Dataset structure looks good!")
    else:
        print("\n❌ Dataset structure needs fixing!")
        print("\nRequired structure:")
        print("LAMP images/")
        print("├── training/")
        print("│   ├── images/    (your training images)")
        print("│   └── labels/    (corresponding .txt files)")
        print("└── validation/")
        print("    ├── images/    (your validation images)")
        print("    └── labels/    (corresponding .txt files)")
    
    return all_good

def run_training_with_error_handling():
    """
    Run training with better error handling
    """
    # Step 1: Create corrected data.yaml
    yaml_file = create_correct_data_yaml()
    if yaml_file is None:
        return
    
    # Step 2: Check dataset structure
    base_path = r"D:\yolo Ryan\PlosOne_publication_codes\LAMP images"
    if not check_dataset_structure(base_path):
        print("\n❌ Please fix the dataset structure before training!")
        return
    
    # Step 3: Initialize YOLO model
    print("\n" + "="*50)
    print("INITIALIZING YOLO MODEL")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Try to download and load YOLOv8
        model = YOLO("yolov8n.pt")  # This will download if not present
        print("✓ YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Step 4: Start training
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    train_config = {
        'data': yaml_file,
        'epochs': 100,  # Reduced for initial test
        'imgsz': 640,
        'batch': 8,     # Reduced batch size to avoid memory issues
        'device': device,
        'workers': 2,   # Reduced workers
        'project': 'lamp_detection',
        'name': 'run1',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'save_period': 20,
        'patience': 30,
    }
    
    try:
        print("Training configuration:")
        for key, value in train_config.items():
            print(f"  {key}: {value}")
        
        results = model.train(**train_config)
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best model saved: runs/detect/run1/weights/best.pt")
        
        # Run validation
        val_results = model.val()
        print("✓ Validation completed")
        
        return model
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Reduce batch size further (try batch=4 or batch=2)")
        print("2. Check if you have enough disk space")
        print("3. Make sure your labels are in correct YOLO format")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 YOLO LAMP Detection Training Setup")
    print("="*50)
    
    # Run the complete setup and training
    model = run_training_with_error_handling()
    
    if model:
        print("\n✅ Setup and training completed successfully!")
        print("Your trained model is ready to use!")
    else:
        print("\n❌ Setup or training failed. Please check the error messages above.")

# Run this script to start!
print("\n" + "="*50)
print("Ready to run! Execute this script in Spyder.")
print("="*50)