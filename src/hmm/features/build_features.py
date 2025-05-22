import os
import pandas as pd
import re
import numpy as np

# Paths
PROCESSED_DATA_PATH = "data/processed/hmm/"

# Load train dataset
def load_train_data(filepath):
    return pd.read_pickle(filepath)

# Group training sentences by sentiment label
def separate_sentiment(df):
    return df[df["category"] == 1]["text"].tolist(), df[df["category"] == -1]["text"].tolist(), \
        df[df["category"] == 0]["text"].tolist()

# Preprocessing: Tokenize a text into list of lowercase alphabetical characters
def tokenize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()

# Apply tokenization for list of text
def tokenize_list_text(lst):
    return [tokenize_text(text) for text in lst]

def main():
    df_train = load_train_data(os.path.join(PROCESSED_DATA_PATH, "train_dataset.pkl"))
    train_pos, train_neg, train_neu = separate_sentiment(df_train)

    processed_train_pos = tokenize_list_text(train_pos)
    processed_train_neg = tokenize_list_text(train_neg)
    processed_train_neu = tokenize_list_text(train_neu)

    np.save(os.path.join(PROCESSED_DATA_PATH, "processed_pos.npy"), np.array(processed_train_pos, dtype=object))
    np.save(os.path.join(PROCESSED_DATA_PATH, "processed_neg.npy"), np.array(processed_train_neg, dtype=object))
    np.save(os.path.join(PROCESSED_DATA_PATH, "processed_neu.npy"), np.array(processed_train_neu, dtype=object))

if __name__ == "__main__":
    main()
