import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Paths
DECISION_TREE_PROCESSED_DATA_PATH = "data/processed/decision_tree/"

def evaluate_model(prediction_path, y_test, output_path, image_path):
    # Load predicted labels
    df = pd.read_csv(prediction_path)
    y_pred = df['predicted_label'].to_numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Ensure experiments directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save evaluation results
    output_file = os.path.join(output_path, "evaluation_results.txt")

    with open(output_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf_matrix))
        f.write("\n\nClassification Report:\n")
        f.write(class_report)

    print("Evaluation metrics saved to:", output_file)

    # Normalize confusion matrix by row (true label count)
    conf_matrix_ratio = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix_ratio, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")

    # Save plot
    os.makedirs(image_path, exist_ok=True)
    plot_file = os.path.join(image_path, "confusion_matrix_normalized.png")
    plt.savefig(plot_file)
    plt.close()

    print("Normalized confusion matrix saved to:", plot_file)
