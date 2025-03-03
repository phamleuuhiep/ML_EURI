import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ChestXRayDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, dataset_type="train"):
        """
        Args:
            img_dir (str or Path): Path to the directory containing images.
            csv_path (str or Path): Path to the CSV file with labels.
            transform (torchvision.transforms): Image transformations.
            dataset_type (str): Type of dataset ('train' or 'test').
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Dataset_type"] == dataset_type]
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["X_ray_image_name"]
        label = self.df.iloc[idx]["Label"]
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB")

        label_dict = {"Normal": 0, "Pnemonia": 1}
        label = label_dict.get(label, -1)

        if self.transform:
            image = self.transform(image)
        
        return image, label

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

training_folder_path = "path/to/train_images"
testing_folder_path = "path/to/test_images"
csv_file_path = "path/to/Chest_xray_Corona_Metadata.csv"

train_dataset = ChestXRayDataset(img_dir=training_folder_path,
                                 csv_path=csv_file_path,
                                 transform=train_transforms,
                                 dataset_type="TRAIN")

test_dataset = ChestXRayDataset(img_dir=testing_folder_path,
                                csv_path=csv_file_path,
                                transform=test_transforms,
                                dataset_type="TEST")

BATCH_SIZE = 16
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

model = MyCNN(num_classes=3)
print(model(torch.randn((16, 3, 224, 224))).shape)
