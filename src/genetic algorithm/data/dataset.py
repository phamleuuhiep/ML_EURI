import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, dataset_type="TRAIN", max_samples=None):
        self.img_dir = img_dir
        self.transform = transform
        self.metadata = pd.read_csv(csv_path)
        self.metadata = self.metadata[self.metadata['Dataset_type'] == dataset_type]
        if max_samples is not None:
            self.metadata = self.metadata[:max_samples]
        self.image_paths = [os.path.join(img_dir, img_name) for img_name in self.metadata['X_ray_image_name']]
        self.labels = self.metadata['Label'].apply(lambda x: 1 if x == 'Pnemonia' else 0).values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])