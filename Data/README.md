## Data Structure

The `Data` folder contains the raw images used for training, validation, and testing the LAMP image classification model. The structure is as follows:

```
Data/
├── training/
│   └── [training images]
├── validation/
│   └── [validation images]
└── testing/
    └── [testing images]
```

To use these datasets:
1. Download the images
2. Place the images in their respective folders (training, validation, testing)
3. Update the `data.yaml` file to reflect the paths to these folders
