import os
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Directory paths
PROCESSED_DIR = "data/processed/naive_bayes/feature/"
TRAINED_DIR = "models/experiments/naive_bayes/"

# Ensure output directory exists
os.makedirs(TRAINED_DIR, exist_ok=True)

# Load preprocessed data
X_train_pca = np.load(os.path.join(PROCESSED_DIR, "train_features.npy"))
train_labels = np.load(os.path.join(PROCESSED_DIR, "train_labels.npy"))

# Initialize and train Naive Bayes model
clf = GaussianNB(var_smoothing=1e-9) # Prevents division by zero in probability calculations by adding a tiny value to variances.
clf.fit(X_train_pca, train_labels)

# Save trained model
model_filename = os.path.join(TRAINED_DIR, "naive_bayes_model.pkl")
joblib.dump(clf, model_filename)

# Print model summary
print("Naive Bayes Model Summary:")
print(f"Number of features: {X_train_pca.shape[1]}")
print(f"Number of classes: {len(clf.classes_)}")
print("Class probabilities:", clf.class_prior_)