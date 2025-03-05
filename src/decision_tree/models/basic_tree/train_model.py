import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Paths
PROCESSED_DATA_PATH = "data/processed/decision_tree/"
TRAINED_MODEL_PATH = "models/trained/decision_tree/basic_tree/"

# Ensure output directory exists
os.makedirs(TRAINED_MODEL_PATH, exist_ok=True)

# Load preprocessed data
X_train = np.load(os.path.join(PROCESSED_DATA_PATH, "X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DATA_PATH, "y_train.npy"))

# Initialize and train Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save trained model
model_filename = os.path.join(TRAINED_MODEL_PATH, "decision_tree_model.pkl")
joblib.dump(clf, model_filename)

print(f"Model trained and saved at {model_filename}")

# Print model summary
print("Decision Tree Model Summary:")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {clf.n_classes_}")
print(f"Tree depth: {clf.get_depth()}")
print(f"Number of leaves: {clf.get_n_leaves()}")
print("Feature importances:", clf.feature_importances_)
