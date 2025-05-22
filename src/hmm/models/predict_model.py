import joblib
import os
import numpy as np
import pandas as pd
from ..visualization.visualize import evaluate_model
from ..features.build_features import tokenize_text
from ..models.train_model import encode_text, load_processed_data, build_vocab

# Path
PROCESSED_DATA_PATH = "data/processed/hmm/"
EXPERIMENTS_PATH = "models/experiments/hmm/"
IMAGE_PATH = "reports/figures/hmm/"

def predict(text, models, vocab, unk_id):
    seq = tokenize_text(text)
    encoded = encode_text(seq, vocab, unk_id)
    scores = {}
    for label, model in models.items():
        obs = np.array(encoded).astype(int).reshape(-1, 1)
        if len(obs) == 0:
            scores[label] = float('-inf')  # Assign very low score if sequence is empty
        else:
            scores[label] = model.score(obs)
    return max(scores, key=scores.get)

def predict_testset(testset, output_path, models, vocab, unk_id):
    prediction = [predict(text, models, vocab, unk_id) for text in testset]

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
    processed_train_pos = load_processed_data(os.path.join(PROCESSED_DATA_PATH, "processed_pos.npy"))
    processed_train_neg = load_processed_data(os.path.join(PROCESSED_DATA_PATH, "processed_neg.npy"))
    processed_train_neu = load_processed_data(os.path.join(PROCESSED_DATA_PATH, "processed_neu.npy"))

    vocab, unk_id = build_vocab(processed_train_pos + processed_train_neg + processed_train_neu)

    model_pos = joblib.load(os.path.join(EXPERIMENTS_PATH, "model_pos.pkl"))
    model_neg = joblib.load(os.path.join(EXPERIMENTS_PATH, "model_neg.pkl"))
    model_neu = joblib.load(os.path.join(EXPERIMENTS_PATH, "model_neu.pkl"))

    models = {
        0: model_neu, # neutral
        1: model_pos, # positive
        -1: model_neg # negative
    }

    df_test = pd.read_pickle(os.path.join(PROCESSED_DATA_PATH, "test_dataset.pkl"))
    predict_file = predict_testset(df_test["text"].tolist(), EXPERIMENTS_PATH, models, vocab, unk_id)
    evaluate_model(predict_file, df_test["category"].to_numpy(), EXPERIMENTS_PATH, IMAGE_PATH)

if __name__ == "__main__":
    main()
