import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
TWITTER_RAW_DATA_PATH = "data/raw/dataset_phase_2/Twitter_Data.csv"
REDDIT_RAW_DATA_PATH = "data/raw/dataset_phase_2/Reddit_Data.csv"
TRAIN_DATA_PATH = "data/processed/hmm/train_dataset.pkl"
TEST_DATA_PATH = "data/processed/hmm/test_dataset.pkl"

# Load and clean raw dataset into data frames
def load_and_clean_dataset(path, text_column_name):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains('Unnamed: 0')]
    df = df.rename(columns={text_column_name: "text"})
    df.dropna(subset=["text", "category"], inplace=True)
    return df

# Split the dataset into training set and test set
def split_dataset(df):
    return train_test_split(
        df, 
        test_size = 0.2,          
        stratify = df["category"],  
        random_state = 42         
    )

def main():
    df_twitter = load_and_clean_dataset(TWITTER_RAW_DATA_PATH, "clean_text")
    df_reddit = load_and_clean_dataset(REDDIT_RAW_DATA_PATH, "clean_comment")

    # Combine into a dataset for both sources
    df = pd.concat([df_twitter, df_reddit], ignore_index=True)

    # Split into train and test dataset
    df_train, df_test = split_dataset(df)
    df_train.to_pickle(TRAIN_DATA_PATH)
    df_test.to_pickle(TEST_DATA_PATH)

if __name__ == "__main__":
    main()
