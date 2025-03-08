import matplotlib.pyplot as plt
from data.dataset import ChestXRayDataset, train_transforms, test_transforms
from genetic_algorithm import genetic_algorithm
from pathlib import Path


# Define base path
base_path = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'raw'
# Define paths
training_folder_path = base_path / 'Coronahack-Chest-XRay-Dataset' / 'train'
testing_folder_path = base_path / 'Coronahack-Chest-XRay-Dataset' / 'test'
csv_path = base_path / 'Chest_xray_Corona_Metadata.csv'

train_dataset = ChestXRayDataset(img_dir=training_folder_path, csv_path=csv_path, transform=train_transforms, dataset_type="TRAIN", max_samples=4000)
test_dataset = ChestXRayDataset(img_dir=testing_folder_path, csv_path=csv_path, transform=test_transforms, dataset_type="TEST", max_samples=1000)

best_hparams = genetic_algorithm(train_dataset, test_dataset, population_size=5, generations=4, mutation_rate=0.1)
print("Best hyperparameters found:", best_hparams)