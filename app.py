from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords

# --- 1. Setup & Model Loading ---
app = FastAPI(title="SVM Sentiment Analysis Engine")

# Load the best model we just trained
# The pipeline includes the TfidfVectorizer and StandardScaler, 
# so we don't need to manually transform the input text!
model = joblib.load('./models/linear_svm_model.pkl')

# --- 2. Preprocessing Logic ---
# The model expects 'clean_text' and the 4 numerical features
# We need to replicate the cleaning we did in preprocess.py
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

class SentimentRequest(BaseModel):
    text: str

def get_meta_features(text: str):
    return {
        "word_count": len(text.split()),
        "char_count": len(text),
        "exclamation_count": text.count('!'),
        "question_count": text.count('?')
    }

def clean_input_text(text: str):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# --- 3. API Endpoints ---

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is Running"}

@app.post("/predict")
def predict(request: SentimentRequest):
    raw_text = request.text
    
    # 1. Extract Meta Features
    meta = get_meta_features(raw_text)
    
    # 2. Clean Text
    cleaned = clean_input_text(raw_text)
    
    # 3. Create a DataFrame match the training format
    import pandas as pd
    input_df = pd.DataFrame([{
        'clean_text': cleaned,
        'word_count': meta['word_count'],
        'char_count': meta['char_count'],
        'exclamation_count': meta['exclamation_count'],
        'question_count': meta['question_count']
    }])
    
    # 4. Predict
    prediction = model.predict(input_df)[0]
    label = "Positive" if prediction == 1 else "Negative"
    
    # Get probability/confidence if using LinearSVC (requires extra steps)
    # Since LinearSVC doesn't do predict_proba by default, we just return the label.
    
    return {
        "text": raw_text,
        "sentiment": label,
        "meta_features": meta
    }
