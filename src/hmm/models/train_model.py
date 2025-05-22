import os
import joblib
import numpy as np
from collections import Counter
from hmmlearn.hmm import CategoricalHMM

# Paths
PROCESSED_DATA_PATH = "data/processed/hmm/"
TRAINED_MODEL_PATH = "models/experiments/hmm/"

unk_token = "<UNK>"

# Load processed data
def load_processed_data(filepath):
    return np.load(filepath, allow_pickle=True).tolist()

# Build common vocabulary for 3 models
def build_vocab(text_list):
    freqs = Counter([token for text in text_list for token in text])

    word_to_id = {}
    word_id = 0
    
    # Add frequent words to vocab (freq > 4)
    for word, freq in freqs.items():
        if freq > 4:
            word_to_id[word] = word_id
            word_id += 1
    
    # Add <UNK> token - representing for rare tokens (freq <= 4)
    word_to_id[unk_token] = word_id

    return word_to_id, word_id

# Encode tokens to integers
def encode_text(text, vocab, unk_id):
    return [vocab.get(token, unk_id) for token in text]

# Encode a list of sequences of tokens
def encode(text_list, vocab, unk_id):
    return [encode_text(text, vocab, unk_id) for text in text_list]

# Train HMM model
def train_model(encoded_list, n_features, n_states=5):
    encoded_list = [seq for seq in encoded_list if len(seq) > 0]
    if len(encoded_list) == 0:
        raise ValueError("No non-empty sequences available for training.")
    flatten = np.concatenate(encoded_list).astype(int).reshape(-1, 1)
    text_lens = [len(text) for text in encoded_list]

    model = CategoricalHMM(n_components=n_states, n_iter=500, tol=20.0, verbose=True)
    model.n_features = n_features
    model.fit(flatten, text_lens)

    return model

def main():
    processed_train_pos = load_processed_data(os.path.join(PROCESSED_DATA_PATH, "processed_pos.npy"))
    processed_train_neg = load_processed_data(os.path.join(PROCESSED_DATA_PATH, "processed_neg.npy"))
    processed_train_neu = load_processed_data(os.path.join(PROCESSED_DATA_PATH, "processed_neu.npy"))

    vocab, unk_id = build_vocab(processed_train_pos + processed_train_neg + processed_train_neu)
    encoded_train_pos = encode(processed_train_pos, vocab, unk_id)
    encoded_train_neg = encode(processed_train_neg, vocab, unk_id)
    encoded_train_neu = encode(processed_train_neu, vocab, unk_id)

    model_pos = train_model(encoded_train_pos, len(vocab))
    model_neg = train_model(encoded_train_neg, len(vocab))
    model_neu = train_model(encoded_train_neu, len(vocab))

    joblib.dump(model_pos, os.path.join(TRAINED_MODEL_PATH, "model_pos.pkl"))
    joblib.dump(model_neg, os.path.join(TRAINED_MODEL_PATH, "model_neg.pkl"))
    joblib.dump(model_neu, os.path.join(TRAINED_MODEL_PATH, "model_neu.pkl"))

if __name__ == "__main__":
    main()