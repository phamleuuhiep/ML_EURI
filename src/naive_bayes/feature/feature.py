import os
import numpy as np
from PIL import Image
import shutil

# Directory paths
OUTPUT_DATA_DIR = "data/processed/naive_bayes/data/"
PROCESSED_DIR = "data/processed/naive_bayes/feature/"
IMAGE_DIMENSIONS = (128, 128)

# Retrieve stored file paths
def retrieve_file_paths(filename):
    return np.load(os.path.join(OUTPUT_DATA_DIR, filename))

# Convert images: resize, grayscale, and normalize
def transform_images(file_paths):
    processed_images = []
    for path in file_paths:
        image = Image.open(path).convert("L")  # Convert to grayscale
        image = image.resize(IMAGE_DIMENSIONS)
        image = np.array(image) / 255.0  # Normalize pixel values
        processed_images.append(image.flatten())  # Flatten for model input
    return np.array(processed_images)

# Store the transformed images
def store_transformed_data(data, filename):
    np.save(os.path.join(PROCESSED_DIR, filename), data)

# Copy label files without modification
def copy_label_files():
    for label_file in ["train_labels.npy", "test_labels.npy"]:
        src_path = os.path.join(OUTPUT_DATA_DIR, label_file)
        dest_path = os.path.join(PROCESSED_DIR, label_file)
        if os.path.exists(src_path):  # Check if label file exists before copying
            shutil.copy2(src_path, dest_path)  # Copy label file

# Main processing function
def main():
    for dataset in ["train_images.npy", "test_images.npy"]:
        paths = retrieve_file_paths(dataset)
        transformed_data = transform_images(paths)
        store_transformed_data(transformed_data, dataset)
    # Copy label files
    copy_label_files()

if __name__ == "__main__":
    main()
