from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import numpy as np

# Paths
PROCESSED_DATA_PATH = "data/processed/decision_tree/"
EXPERIMENTS_PATH = "models/experiments/decision_tree/cross_validation/"

# Load preprocessed data
X_train = np.load(os.path.join(PROCESSED_DATA_PATH, "X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DATA_PATH, "y_train.npy"))

# Define batch generator
def batch_generator(X, y, batch_size=1000):
    n_samples = X.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[start:end], y[start:end]

# Custom batch training
param_grid = {
    'max_depth': [5, 10, 20, 30],
    'min_samples_split': [2, 10, 50, 100],
    'min_samples_leaf': [1, 10, 50]
}

dt_model = DecisionTreeClassifier()

grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)

# Fit in batches
for X_batch, y_batch in batch_generator(X_train, y_train):
    grid_search.fit(X_batch, y_batch)

# Print the best parameters and score
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Retrain the final model on the entire training set
final_model = DecisionTreeClassifier(**grid_search.best_params_)
final_model.fit(X_train, y_train)

# Save the final model
joblib.dump(final_model, os.path.join(EXPERIMENTS_PATH, 'decision_tree_model.pkl'))

print("Final model saved as 'decision_tree_model.pkl'.")

