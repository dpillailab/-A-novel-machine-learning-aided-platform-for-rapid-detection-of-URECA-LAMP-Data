
## Image Dataset
The full image dataset (353 images, 664 MB) is stored externally due to size constraints.
Access the images here: [https://uofc-my.sharepoint.com/:f:/g/personal/biniyamkahsay_mezgeb_ucalgary_ca/EvIoFxv3ax5MlwfrVxc7BCQBUH93eAmfBQeKVuffIA32oQ?e=ez8Q1V]

To use these images:
1. Download the entire folder from OneDrive.
2. Place the downloaded images in the `images` directory of this repository.
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
