import joblib
import os
import numpy as np

# Path
DECISION_TREE_PROCESSED_DATA_PATH = "data/processed/decision_tree/"
DATA_FILENAMES_PATH = "data/processed/shared/X_test.npy"

def test_predict(model_path, output_path):
    # Load test data
    X_test = np.load(os.path.join(DECISION_TREE_PROCESSED_DATA_PATH, "X_test.npy"))

    # Load trained model
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Load filenames for test data
    file_paths = np.load(DATA_FILENAMES_PATH)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Create output file
    output_file = os.path.join(output_path, "test_predictions.csv")

    # Save predictions to CSV
    with open(output_file, "w") as f:
        f.write("filename,predicted_label\n")
        for file_path, label in zip(file_paths, y_pred):
            f.write(f"{os.path.basename(file_path)},{label}\n")

    print("Predictions saved to:", output_file)

    return output_file

