import os
import joblib
import numpy as np
from src.naive_bayes.visualize.visualize import evaluate_model

# Directory paths
PROCESSED_DIR = "data/processed/naive_bayes/feature/"
DATA_TEST_DIR = "data/processed/naive_bayes/data/test_images.npy"
MODEL_DIR = "models/experiments/naive_bayes/naive_bayes_model.pkl"
RESULT_DIR = "models/experiments/naive_bayes/"
IMAGE_DIR = "reports/figures/naive_bayes/"

def test_predict(model_dir, result_dir):
    # Load test data
    X_train_pca = np.load(os.path.join(PROCESSED_DIR, "test_features.npy"))
    # Load trained model
    model = joblib.load(model_dir)
    # Make predictions
    predict_labels = model.predict(X_train_pca)
    # Load filenames for test data
    file_dirs = np.load(DATA_TEST_DIR)
    # Ensure the result directory exists
    os.makedirs(result_dir, exist_ok=True)
    # Create result file
    result_file = os.path.join(result_dir, "test_predictions.csv")
    # Save predictions to CSV
    with open(result_file, "w") as f:
        f.write("filename,predicted_label\n")
        for file_dir, label in zip(file_dirs, predict_labels):
            f.write(f"{os.path.basename(file_dir)},{label}\n")
    return result_file

# Run model prediction on test data
predict_file = test_predict(MODEL_DIR, RESULT_DIR)

# Evaluate and visualize model performance
evaluate_model(predict_file, RESULT_DIR, IMAGE_DIR)