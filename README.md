# -A-novel-machine-learning-aided-platform-for-rapid-detection-of-URECA-LAMP-Data
# LAMP Image Classification using YOLOv8

This repository contains the code, trained model, and datasets for classifying LAMP (Loop-mediated Isothermal Amplification) images using YOLOv8. The project focuses on automated detection and classification of LAMP assay results.

## Contents

- `yolo_script.py`: Python script for training YOLOv8 models and running predictions on LAMP images
- `best.pt`: Pre-trained YOLOv8 model for LAMP image classification
- `data/`: 
  - `train/`: Raw training LAMP images
  - `test/`: Test sample LAMP images
- `data.yaml`: Dataset configuration file for YOLOv8

## Requirements

- Python 3.7+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- tqdm

Install the required packages using:

```bash
pip install torch ultralytics opencv-python tqdm
```

## Usage

The `yolo_script.py` provides functionality for both training new models and running predictions using pre-trained models.

### Training

To train a new YOLOv8 model on the LAMP dataset:

```bash
python yolo_script.py --mode train --data path/to/data.yaml --test_folder path/to/test_images --output_folder path/to/output
```

Arguments:
- `--data`: Path to your `data.yaml` file describing the dataset
- `--test_folder`: Directory containing test images
- `--output_folder`: Directory to save output images with predictions

### Prediction

To use the pre-trained model (`best.pt`) for predictions:

```bash
python yolo_script.py --mode predict --model best.pt --test_folder path/to/test_images --output_folder path/to/output
```

Arguments:
- `--model`: Path to the pre-trained model file (e.g., "best.pt")
- `--test_folder`: Directory containing test images
- `--output_folder`: Directory to save output images with predictions

## Model

The `best.pt` file is a pre-trained YOLOv8 model specifically tuned for LAMP image classification. It has been trained on the provided LAMP image dataset and can be used for immediate predictions.

## Dataset

- `data/train/`: Contains raw training LAMP images used to train the model
- `data/test/`: Contains test samples for evaluating the model's performance

The `data.yaml` file describes the dataset structure and class names for the YOLO model.

## Script Details

The `yolo_script.py` includes the following main functions:

1. `train_yolo_model()`: Trains a new YOLOv8 model on the LAMP dataset
2. `load_yolo_model()`: Loads a pre-trained YOLOv8 model
3. `run_yolo_image_test_folder()`: Runs predictions on a folder of test images and saves visualized results

The script uses CUDA if available, otherwise falls back to CPU processing.

## Citation

If you use this code or model in your research, please cite:

Mezgeb, B.K., et al. (2024). [Paper Title]. [Journal Name].

## License

[Choose and include an appropriate license here]

## Contact

Biniyam Kahsay Mezgeb - biniyamkahsay.mezgeb@ucalgary.ca

Project Link: [https://github.com/yourusername/your-repo-name](https://github.com/yourusername/your-repo-name](https://github.com/dpillailab/-A-novel-machine-learning-aided-platform-for-rapid-detection-of-URECA-LAMP-Data).

## Acknowledgments

- Ultralytics for the YOLOv8 implementation
- 
