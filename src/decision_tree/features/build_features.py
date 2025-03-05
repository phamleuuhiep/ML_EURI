import os
import numpy as np
from PIL import Image

# Paths
PROCESSED_DATA_PATH = "data/processed/shared/"
DECISION_TREE_PROCESSED_DATA_PATH = "data/processed/decision_tree/"
IMAGE_SIZE = (128, 128)

# Load file paths
def load_file_paths(filename):
    return np.load(os.path.join(PROCESSED_DATA_PATH, filename))

# Preprocess images (resize, grayscale, normalize)
def preprocess_images(file_paths):
    images = []
    for file_path in file_paths:
        img = Image.open(file_path).convert("L")  # grayscale
        img = img.resize(IMAGE_SIZE)
        img = np.array(img) / 255.0  # normalize to [0, 1]
        images.append(img.flatten())  # flatten for decision tree
    return np.array(images)

# Save preprocessed data
def save_preprocessed_data(X, filename):
    np.save(os.path.join(DECISION_TREE_PROCESSED_DATA_PATH, filename), X)

def main():
    for split in ["X_train.npy", "X_test.npy"]:
        file_paths = load_file_paths(split)
        processed_data = preprocess_images(file_paths)
        save_preprocessed_data(processed_data, split)
    print("Preprocessing completed and saved to:",
          DECISION_TREE_PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()
