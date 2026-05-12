import pandas as pd

print("Loading datasets...")

# ---------------------------------------------------------
# 1. Loading the Twitter Dataset (Sentiment140)
# ---------------------------------------------------------
# We must explicitly define the column names and the encoding
twitter_cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
twitter_data_path = '../data/training.1600000.processed.noemoticon.csv' # Adjust path if needed

try:
    df_twitter = pd.read_csv(twitter_data_path, encoding='latin-1', names=twitter_cols)
    
    # The sentiment labels in this dataset are 0 (Negative) and 4 (Positive).
    # Let's map the 4s to 1s to make it standard binary (0 and 1).
    df_twitter['sentiment'] = df_twitter['sentiment'].replace(4, 1)
    
    # Drop the columns we don't need for the SVM
    df_twitter = df_twitter[['sentiment', 'text']]
    
    print(f"Twitter dataset loaded successfully! Shape: {df_twitter.shape}")
except FileNotFoundError:
    print("Error: Could not find the Twitter dataset. Check your 'data/' folder.")


# ---------------------------------------------------------
# 2. Loading the IMDB Movie Reviews Dataset
# ---------------------------------------------------------
imdb_data_path = '../data/IMDB Dataset.csv' # Adjust path if needed

try:
    df_imdb = pd.read_csv(imdb_data_path)
    
    # IMDB labels are strings: "positive" and "negative". 
    # Machines need numbers, so we map them to 1 and 0.
    df_imdb['sentiment'] = df_imdb['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Rename 'review' column to 'text' so it matches our Twitter dataframe format
    df_imdb.rename(columns={'review': 'text'}, inplace=True)
    
    print(f"IMDB dataset loaded successfully! Shape: {df_imdb.shape}")
except FileNotFoundError:
    print("Error: Could not find the IMDB dataset. Check your 'data/' folder.")