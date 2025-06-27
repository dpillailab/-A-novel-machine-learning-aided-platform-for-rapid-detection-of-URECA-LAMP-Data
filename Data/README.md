
## Image Dataset
The full image dataset (353 images, 664 MB) is stored externally due to size constraints.
Access the images here: [[https://uofc-my.sharepoint.com/:f:/g/personal/biniyamkahsay_mezgeb_ucalgary_ca/EvIoFxv3ax5MlwfrVxc7BCQBUH93eAmfBQeKVuffIA32oQ?e=ez8Q1V](https://uofc-my.sharepoint.com/:f:/g/personal/biniyamkahsay_mezgeb_ucalgary_ca/Eu1Q8Jvfv31Iu5_xcW-H1Q4BulGLO-RJFrWdV07yFeW6oQ?e=dVs0og)]

To use these images:
1. Download the entire folder from OneDrive.
2. Place the downloaded images in the `images` directory of this repository.
## Data Structure

The `LAMP_dataset_PLOsOne` folder contains the raw images used for training, validation, and testing the LAMP image classification model. The structure is as follows:

```
Data/
├── train/
│   └── [images]
├── validation/
│   └── [images]
└── testing/
    └── [images]
```

To use these datasets:
1. Download the images
2. Place the images in their respective folders (training, validation, testing)
3. Update the `data.yaml` file to reflect the paths to these folders
