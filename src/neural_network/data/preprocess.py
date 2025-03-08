import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from neural_network.data.dataset import ChestXRayDataset

# # Define paths
# training_folder_path = Path('C:/Users/ASUS GAMERS/OneDrive/Máy tính/ML_Project_1/ML_EURI/data/raw/Coronahack-Chest-XRay-Dataset/train')
# testing_folder_path = Path('C:/Users/ASUS GAMERS/OneDrive/Máy tính/ML_Project_1/ML_EURI/data/raw/Coronahack-Chest-XRay-Dataset/test')
# csv_path = 'C:/Users/ASUS GAMERS/OneDrive/Máy tính/ML_Project_1/ML_EURI/data/raw/Chest_xray_Corona_Metadata.csv'

# Define base path
base_path = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'raw'
# Define paths
training_folder_path = base_path / 'Coronahack-Chest-XRay-Dataset' / 'train'
testing_folder_path = base_path / 'Coronahack-Chest-XRay-Dataset' / 'test'
csv_path = base_path / 'Chest_xray_Corona_Metadata.csv'

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(size=(224, 224), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])

# Create datasets
train_dataset = ChestXRayDataset(img_dir=training_folder_path, csv_path=csv_path, transform=train_transforms, dataset_type="TRAIN")
test_dataset = ChestXRayDataset(img_dir=testing_folder_path, csv_path=csv_path, transform=test_transforms, dataset_type="TEST")

# Batch size
BATCH_SIZE = 16

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
