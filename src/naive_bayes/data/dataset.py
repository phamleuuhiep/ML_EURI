import os
import pandas as pd
import numpy as np

# Directory paths
SOURCE_DATA_DIR = "data/raw/"
OUTPUT_DATA_DIR = "data/processed/naive_bayes/data/"
INFO_FILE = os.path.join(SOURCE_DATA_DIR, "Chest_xray_Corona_Metadata.csv")

# Read metadata from CSV file
def read_metadata():
    return pd.read_csv(INFO_FILE)

# Process metadata to filter necessary records and assign class labels
def process_and_label_data(metadata):
    selected_data = metadata[
        (metadata["Label"] == "Normal") |
        ((metadata["Label"] == "Pnemonia") & (
            metadata["Label_1_Virus_category"].isin(["Virus", "bacteria"])))
    ].copy()

    category_labels = []
    for _, record in selected_data.iterrows():
        if record["Label"] == "Normal":
            category_labels.append(1)  # Category 1: Normal case
        elif record["Label"] == "Pnemonia" and record["Label_1_Virus_category"] == "Virus":
            category_labels.append(2)  # Category 2: Viral pneumonia
        elif record["Label"] == "Pnemonia" and record["Label_1_Virus_category"] == "bacteria":
            category_labels.append(3)  # Category 3: Bacterial pneumonia

    selected_data["Category"] = category_labels
    return selected_data

# Divide data into training and testing sets based on dataset type
def partition_data(metadata):
    train_set = metadata[metadata["Dataset_type"] == "TRAIN"]
    test_set = metadata[metadata["Dataset_type"] == "TEST"]

    train_images = [os.path.join(SOURCE_DATA_DIR + 'Coronahack-Chest-XRay-Dataset/train/', fname)
                    for fname in train_set["X_ray_image_name"].values]
    train_labels = train_set["Category"].values

    test_images = [os.path.join(SOURCE_DATA_DIR + 'Coronahack-Chest-XRay-Dataset/test/', fname)
                   for fname in test_set["X_ray_image_name"].values]
    test_labels = test_set["Category"].values

    return train_images, test_images, train_labels, test_labels

# Store processed datasets as numpy arrays
def store_partitioned_data(train_images, test_images, train_labels, test_labels):
    np.save(os.path.join(OUTPUT_DATA_DIR, "train_images.npy"), np.array(train_images))
    np.save(os.path.join(OUTPUT_DATA_DIR, "test_images.npy"), np.array(test_images))
    np.save(os.path.join(OUTPUT_DATA_DIR, "train_labels.npy"), np.array(train_labels))
    np.save(os.path.join(OUTPUT_DATA_DIR, "test_labels.npy"), np.array(test_labels))

# Main function to execute data processing workflow
def main():
    metadata = read_metadata()
    refined_data = process_and_label_data(metadata)
    train_images, test_images, train_labels, test_labels = partition_data(refined_data)
    store_partitioned_data(train_images, test_images, train_labels, test_labels)

if __name__ == "__main__":
    main()