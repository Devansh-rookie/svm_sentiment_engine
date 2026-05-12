import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Download stopwords (only needs to run once)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Enable pandas progress apply
tqdm.pandas()

def extract_meta_features(df, text_column='text'):
    """Extracts custom features before the text is stripped of punctuation."""
    print("Extracting meta-features...")
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    df['char_count'] = df[text_column].apply(lambda x: len(str(x)))
    df['exclamation_count'] = df[text_column].apply(lambda x: str(x).count('!'))
    df['question_count'] = df[text_column].apply(lambda x: str(x).count('?'))
    return df

def clean_text(text):
    """Cleans the text for TF-IDF vectorization."""
    text = str(text).lower() # Lowercase
    text = re.sub(r'<[^>]+>', ' ', text) # Remove HTML tags (IMDB)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs (Twitter)
    text = re.sub(r'\@\w+|\#', '', text) # Remove @mentions and hashtags
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation and numbers
    
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)

def main():
    print("--- Starting Preprocessing Pipeline ---")
    
    # 1. Load Data
    print("Loading raw data...")
    twitter_cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    
    # PRO TIP FOR YOUR MACBOOK: 
    # Processing 1.6M rows will take 10-15 minutes. 
    # Let's take a balanced random sample of 100,000 tweets for rapid development.
    # We can scale back up to 1.6M for the final training run.
    df_twitter = pd.read_csv('./data/training.1600000.processed.noemoticon.csv', encoding='latin-1', names=twitter_cols)
    df_twitter = df_twitter.sample(n=100000, random_state=42).reset_index(drop=True)
    df_twitter['sentiment'] = df_twitter['sentiment'].replace(4, 1)
    df_twitter = df_twitter[['sentiment', 'text']]
    
    df_imdb = pd.read_csv('./data/IMDB Dataset.csv')
    df_imdb['sentiment'] = df_imdb['sentiment'].map({'positive': 1, 'negative': 0})
    df_imdb.rename(columns={'review': 'text'}, inplace=True)

    # 2. Extract Meta-Features (BEFORE cleaning)
    print("\nProcessing Twitter Data...")
    df_twitter = extract_meta_features(df_twitter)
    print("Cleaning Twitter text...")
    df_twitter['clean_text'] = df_twitter['text'].progress_apply(clean_text)

    print("\nProcessing IMDB Data...")
    df_imdb = extract_meta_features(df_imdb)
    print("Cleaning IMDB text...")
    df_imdb['clean_text'] = df_imdb['text'].progress_apply(clean_text)

    # 3. Save the Processed Data
    print("\nSaving processed datasets...")
    df_twitter.to_csv('./data/twitter_processed.csv', index=False)
    df_imdb.to_csv('./data/imdb_processed.csv', index=False)
    print("Done! Data is ready for the SVM.")

if __name__ == "__main__":
    main()