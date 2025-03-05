import os
import pandas as pd
import numpy as np

# Paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/shared/"
METADATA_FILE = os.path.join(RAW_DATA_PATH, "Chest_xray_Corona_Metadata.csv")

# Load metadata
def load_metadata():
    return pd.read_csv(METADATA_FILE)

# Assign class labels and filter relevant records
def filter_and_assign_classes(metadata):
    filtered_data = metadata[
        (metadata["Label"] == "Normal") |
        ((metadata["Label"] == "Pnemonia") & (
            metadata["Label_1_Virus_category"].isin(["Virus", "bacteria"])))
    ].copy()

    classes = []
    for _, row in filtered_data.iterrows():
        if row["Label"] == "Normal":
            classes.append(1)  # Class 1: Normal
        elif row["Label"] == "Pnemonia" and row["Label_1_Virus_category"] == "Virus":
            classes.append(2)  # Class 2: Viral pneumonia
        elif row["Label"] == "Pnemonia" and row["Label_1_Virus_category"] == "bacteria":
            classes.append(3)  # Class 3: Bacterial pneumonia

    filtered_data["Class"] = classes
    return filtered_data

# Split data based on Dataset_type (TRAIN/TEST)
def split_data_by_type(metadata):
    train_data = metadata[metadata["Dataset_type"] == "TRAIN"]
    test_data = metadata[metadata["Dataset_type"] == "TEST"]

    X_train = [os.path.join(RAW_DATA_PATH + 'Coronahack-Chest-XRay-Dataset/train/', fname)
               for fname in train_data["X_ray_image_name"].values]
    y_train = train_data["Class"].values

    X_test = [os.path.join(RAW_DATA_PATH + 'Coronahack-Chest-XRay-Dataset/test/', fname)
              for fname in test_data["X_ray_image_name"].values]
    y_test = test_data["Class"].values

    return X_train, X_test, y_train, y_test

# Save data
def save_split_data(X_train, X_test, y_train, y_test):
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_train.npy"), np.array(X_train))
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_test.npy"), np.array(X_test))
    np.save(os.path.join(PROCESSED_DATA_PATH, "y_train.npy"), np.array(y_train))
    np.save(os.path.join(PROCESSED_DATA_PATH, "y_test.npy"), np.array(y_test))

def main():
    metadata = load_metadata()
    filtered_metadata = filter_and_assign_classes(metadata)
    X_train, X_test, y_train, y_test = split_data_by_type(filtered_metadata)
    save_split_data(X_train, X_test, y_train, y_test)
    print("Filtered data split and saved to:", PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()
