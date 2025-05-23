import joblib
import os
import pandas as pd
from pgmpy.inference import VariableElimination
from ..visualization.visualize import evaluate_model
from ..features.build_features import encode_train_data, cluster_sentences

# Path
PROCESSED_DATA_PATH = "data/processed/bayesian_network/"
EXPERIMENTS_PATH = "models/experiments/bayesian_network/"
IMAGE_PATH = "reports/figures/bayesian_network/"

def predict(text, inference):
    # evidence: dict of node_name -> cluster_id from preprocessing
    evidence = {
        'Node_1': text[0],
        'Node_2': text[1],
        'Node_3': text[2],
        'Node_4': text[3],
        'Node_5': text[4],
        'Node_6': text[5]
    }

    # Query the most probable label given evidence
    predict = inference.map_query(variables=['Label'], evidence=evidence, show_progress=False)
    return predict['Label']

def predict_testset(df_test, output_path, model):
    # Encode test data
    test_embeddings = encode_train_data(df_test)
    test_discrete_features = cluster_sentences(test_embeddings)

    inference = VariableElimination(model)

    prediction = [predict(tc, inference) for tc in test_discrete_features.tolist()]

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Create output file
    output_file = os.path.join(output_path, "test_predictions.csv")

    # Save predictions to CSV
    with open(output_file, "w") as f:
        f.write("predicted_label\n")
        for label in prediction:
            f.write(f"{label}\n")

    print("Predictions saved to:", output_file)

    return output_file

def main():
    model = joblib.load(os.path.join(EXPERIMENTS_PATH, "model.pkl"))

    df_test = pd.read_pickle(os.path.join(PROCESSED_DATA_PATH, "test_dataset.pkl"))

    predict_file = predict_testset(df_test, EXPERIMENTS_PATH, model)
    evaluate_model(predict_file, df_test["category"].to_numpy(), EXPERIMENTS_PATH, IMAGE_PATH)

if __name__ == "__main__":
    main()
