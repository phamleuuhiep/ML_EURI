import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

# Directory paths
PROCESSED_DIR = "data/processed/naive_bayes/feature/"
IMAGE_DIR = "reports/figures/naive_bayes/"

# Load processed image data and labels
def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "train_images.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "test_images.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "train_labels.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "test_labels.npy"))
    return X_train, X_test, y_train, y_test

# Compute and visualize feature correlations
def generate_feature_correlation_heatmap(X_train, top_n=1000):
    variances = pd.DataFrame(X_train).var()
    top_features = variances.nlargest(top_n).index
    corr_matrix = pd.DataFrame(X_train).iloc[:, top_features].corr()

    # Generate and save the clustered heatmap
    sns.clustermap(corr_matrix, cmap="coolwarm", figsize=(12, 10), method="ward")
    plt.title("Clustered Feature Correlation Heatmap (Top 1000 Features)")
    os.makedirs(IMAGE_DIR, exist_ok=True)  # Ensure directory exists
    plt.savefig(os.path.join(IMAGE_DIR, "clustered_feature_correlation_heatmap.png"))

# Remove highly correlated features
def remove_highly_correlated_features(X, threshold=0.9):
    corr_matrix = np.corrcoef(X, rowvar=False)
    upper_tri = np.triu(corr_matrix, k=1)
    to_drop = set(np.where(upper_tri > threshold)[1])
    kept_indices = sorted(set(range(X.shape[1])) - to_drop)
    return X[:, kept_indices], kept_indices

# Apply PCA for dimensionality reduction
def apply_pca(X, n_components=0.99):
    n_features = X.shape[1]
    if n_features < 2:
        print("Skipping PCA: Too few features after selection.")
        return X, None  # Return unmodified data

    n_components = max(1, min(n_features - 1, int(n_features * n_components)))
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca

# Feature selection and dimensionality reduction pipeline
def main():
    X_train, X_test, y_train, y_test = load_data()

    # Step 1: Compute mutual information scores
    mi_scores = mutual_info_classif(X_train, y_train, discrete_features=False, random_state=42, n_jobs=-1)
    selected_feature_indices = np.argsort(mi_scores)[-1000:]

    # Step 2: Apply feature selection
    X_train_selected = X_train[:, selected_feature_indices]
    X_test_selected = X_test[:, selected_feature_indices]

    # Step 3: Remove highly correlated features
    X_train_reduced, reduced_indices = remove_highly_correlated_features(X_train_selected)
    X_test_reduced = X_test_selected[:, reduced_indices]

    # Step 4: Apply PCA
    X_train_pca, pca_model = apply_pca(X_train_reduced)
    X_test_pca = pca_model.transform(X_test_reduced) if pca_model else X_test_reduced

    # Save the transformed datasets
    np.save(os.path.join(PROCESSED_DIR, "train_features.npy"), X_train_pca)
    np.save(os.path.join(PROCESSED_DIR, "test_features.npy"), X_test_pca)

    # Generate heatmap
    generate_feature_correlation_heatmap(X_train)

if __name__ == "__main__":
    main()
