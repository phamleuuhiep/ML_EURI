import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Paths
DECISION_TREE_PROCESSED_DATA_PATH = "data/processed/decision_tree/"
EXPERIMENTS_PATH = "models/experiments/decision_tree/post-pruning/"
MODEL_PATH = "models/experiments/decision_tree/basic_tree/decision_tree_model.pkl"
GRAPH_OUTPUT_PATH = "reports/figures/decision_tree/post-pruning/"

# Load data
X_train = np.load(os.path.join(DECISION_TREE_PROCESSED_DATA_PATH, "X_train.npy"))
y_train = np.load(os.path.join(DECISION_TREE_PROCESSED_DATA_PATH, "y_train.npy"))
X_test = np.load(os.path.join(DECISION_TREE_PROCESSED_DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DECISION_TREE_PROCESSED_DATA_PATH, "y_test.npy"))

# Load the initial decision tree to get pruning path
initial_tree = joblib.load(MODEL_PATH)
ccp_alphas = np.unique(initial_tree.cost_complexity_pruning_path(X_train, y_train)["ccp_alphas"])
print(ccp_alphas)

# Track best parameters
best_alpha = None
best_score = 0
best_model = None
results = []

# Post-pruning 
for ccp_alpha in ccp_alphas:
    pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    pruned_tree.fit(X_train, y_train)
    y_test_pred = pruned_tree.predict(X_test)
    val_accuracy = accuracy_score(y_test, y_test_pred)

    y_train_pred = pruned_tree.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    results.append((ccp_alpha, train_accuracy, val_accuracy))
    
    if val_accuracy > best_score:
        best_score = val_accuracy
        best_alpha = ccp_alpha
        best_model = pruned_tree
    elif val_accuracy == best_score:
        best_alpha = ccp_alpha
        best_model = pruned_tree

# Save results
os.makedirs(EXPERIMENTS_PATH, exist_ok=True)
output_file = os.path.join(EXPERIMENTS_PATH, "pruning_results.csv")

results_df = pd.DataFrame(results, columns=['ccp_alpha', 'train_accuracy', 'val_accuracy'])
results_df.to_csv(output_file, index=False)

# Plot accuracy vs ccp_alpha
plt.figure(figsize=(10, 6))
plt.plot(results_df['ccp_alpha'], results_df['train_accuracy'], marker='o', label='Training Accuracy')
plt.plot(results_df['ccp_alpha'], results_df['val_accuracy'], marker='x', label='Validation Accuracy')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs ccp_alpha')
plt.legend()
plt.grid(True)
# Save plot
plot_file = os.path.join(GRAPH_OUTPUT_PATH, "accuracy_vs_ccp_alpha.png")
plt.savefig(plot_file)
plt.show()

# Save the final pruned model
final_model_file = os.path.join(EXPERIMENTS_PATH, "final_pruned_tree.pkl")
pd.to_pickle(best_model, final_model_file)

print("Post-pruning results saved to:", output_file)
print("Best pruning parameter (ccp_alpha):", best_alpha)
print("Final pruned model saved to:", final_model_file)
print("Number of nodes in the final pruned tree:", best_model.tree_.node_count)
print("Maximum depth of the final pruned tree:", best_model.tree_.max_depth)
