import pandas as pd
from tqdm import tqdm
import numpy as np

import os
import shutil
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
from torchvision.transforms import v2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.datasets import ImageFolder
import random

from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np


# Handling images
from PIL import Image
import matplotlib.pyplot as plt

class ChestXRayDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None,dataset_type="train"):
        """
        Args:
        img_dir (str): Path to the directory containing images.
        csv_path (str): Path to the CSV file with labels.
        transform (torchvision.transforms): Image transformations.
        dataset_type (str): Type of dataset to load ("train", "test").
        """
        # Load the CSV file
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Dataset_type"] == dataset_type]
        # Store image directory path
        self.img_dir = img_dir

        # Image transformations
        self.transform = transform

    def __len__(self):
      return len(self.df)

    def __getitem__(self,idx):
        img_name = self.df.iloc[idx]["X_ray_image_name"]
        label = self.df.iloc[idx]["Label"]

        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB")

        label_dict = {"Normal": 0, "Pnemonia": 1}
        label = label_dict.get(label, -1)

        if self.transform:
            image = self.transform(image)

        # image = Image.open(self.images[idx]).convert("RGB")
        # if self.transform:
        #     image = self.transform(image)
        # label = self.labels[idx]
        return image, label
    

