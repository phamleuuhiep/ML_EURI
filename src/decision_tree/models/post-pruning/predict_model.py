from ..predict_model import test_predict
from src.decision_tree.visualization.visualize import evaluate_model

# Paths
MODEL_PATH = "models/experiments/decision_tree/post-pruning/final_pruned_tree.pkl"
EXPERIMENTS_PATH = "models/experiments/decision_tree/post-pruning/"
IMAGE_PATH = "reports/figures/decision_tree/post-pruning/"

predict_file = test_predict(MODEL_PATH, EXPERIMENTS_PATH)
evaluate_model(predict_file, EXPERIMENTS_PATH, IMAGE_PATH)