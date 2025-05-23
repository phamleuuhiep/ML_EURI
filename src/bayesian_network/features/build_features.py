import os
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Paths
PROCESSED_DATA_PATH = "data/processed/bayesian_network/"

# Load train dataset
def load_train_data(filepath):
    return pd.read_pickle(filepath)

# Clean the text - Remove punctuation
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

def encode_train_data(df_train):
  # Load a pre-trained sentence embedding model
  process_model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight and fast
  cleaned_sentences = df_train['text'].apply(clean_text).tolist()
  return process_model.encode(cleaned_sentences, convert_to_numpy=True)

def cluster_sentences(embeddings, k_clusters=5, n_chunks=6):
  embedding_dim = embeddings.shape[1]  # should be 384
  chunk_size = embedding_dim // n_chunks

  # For each chunk, fit KMeans and discretize
  discrete_features = []

  for i in range(n_chunks):
      chunk = embeddings[:, i*chunk_size : (i+1)*chunk_size]
      kmeans = KMeans(n_clusters=k_clusters, random_state=42)
      cluster_ids = kmeans.fit_predict(chunk)
      discrete_features.append(cluster_ids)

  # Transpose and stack to shape (n_samples, n_chunks)
  return np.array(discrete_features).T

def create_df_discrete(df_train, discrete_features, n_chunks=6):
  # Create columns: Node_1, Node_2, ..., Node_n
  node_columns = [f'Node_{i+1}' for i in range(n_chunks)]
  df_discrete = pd.DataFrame(discrete_features, columns=node_columns)

  # Add the label column
  df_discrete['Label'] = df_train['category'].values

  return df_discrete

def main():
    df_train = load_train_data(os.path.join(PROCESSED_DATA_PATH, "train_dataset.pkl"))
    
    sentence_embeddings = encode_train_data(df_train)

    discrete_features = cluster_sentences(sentence_embeddings)

    df_discrete = create_df_discrete(df_train, discrete_features)

    df_discrete.to_pickle(os.path.join(PROCESSED_DATA_PATH, "df_discrete.pkl"))

if __name__ == "__main__":
    main()
