# ELiTUS

A Lightweight Framework for Arbitrary Scale Super-Resolution in Ultrasound Imaging

## Preparing Datasets

The dataset must be organized in a specific structure for the model to correctly load images and masks during training, validation, and testing.

### Expected Folder Structure
```
datasets/
└── US_Data/
    ├──train/
       ├──HR/
       │ ├── image_1.png
       │ ├── image_2.png
       │ └── ...
       ├── LR/
       │ ├──x2.0
       │    ├── image_1.png
       │    └── ...
       │ ├──x1.5_x3.5
       │    ├── image_1.png
       │    └── ...
    ├──val/
    ├──test/
```
- **US_Data/**: Dataset directory.
- **train/**, **val/**, **test/** are the subdirs containing the splits
- **HR/**: Contains the high-resolution ground truth images.
- **LR/**: Contains the Low Resolution couterparts in separate sub-directories organized by scale factor.
---

### Important Notes

- The dataset's name and the subfolders should follow the heirarchy specified in the config.yaml file
  
## Getting Started with Training

Follow these steps to get up and running with the project.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ELiTNet.git
cd ELiTNet
```

### 2. Install UV

Install uv via curl:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This will install uv and set up the environment management system.

### 3. Training the model

Run the training script using uv, which will prompt you to log in to wandb:
```bash
uv run train.py
```
