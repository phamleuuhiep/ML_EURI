import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Directory paths
PROCESSED_DIR = "data/processed/naive_bayes/feature/"

def evaluate_model(model_dir, result_dir, image_dir):
    # Load true labels (test_labels)
    test_labels = np.load(os.path.join(PROCESSED_DIR, "test_labels.npy"))

    # Load predicted labels
    data_frame = pd.read_csv(model_dir)
    predict_labels = data_frame['predicted_label'].to_numpy()

    # Compute evaluation metrics
    accuracyScore = accuracy_score(test_labels, predict_labels)
    confusionMatrix = confusion_matrix(test_labels, predict_labels)
    classificationReport = classification_report(test_labels, predict_labels)

    # Ensure output directory exists
    os.makedirs(result_dir, exist_ok=True)

    # Save evaluation results
    result = os.path.join(result_dir, "evaluation_results.txt")

    with open(result, "w") as f:
        f.write(f"Accuracy: {accuracyScore:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(confusionMatrix))
        f.write("\n\nClassification Report:\n")
        f.write(classificationReport)

    # Normalize confusion matrix by row (true label count)
    confusionMatrix_ratio = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusionMatrix_ratio, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Normal", "Viral Pneumonia", "Bacterial Pneumonia"],
                yticklabels=["Normal", "Viral Pneumonia", "Bacterial Pneumonia"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Naive Bayes - Normalized Confusion Matrix")

    # Save plot
    os.makedirs(image_dir, exist_ok=True)
    plot_file = os.path.join(image_dir, "confusion_matrix_normalized.png")
    plt.savefig(plot_file)
    plt.close()