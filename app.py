# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import re
# import nltk
# from nltk.corpus import stopwords

# # --- 1. Setup & Model Loading ---
# app = FastAPI(title="SVM Sentiment Analysis Engine")

# # Load the best model we just trained
# # The pipeline includes the TfidfVectorizer and StandardScaler, 
# # so we don't need to manually transform the input text!
# model = joblib.load('./models/linear_svm_model.pkl')

# # --- 2. Preprocessing Logic ---
# # The model expects 'clean_text' and the 4 numerical features
# # We need to replicate the cleaning we did in preprocess.py
# nltk.download('stopwords', quiet=True)
# stop_words = set(stopwords.words('english'))

# class SentimentRequest(BaseModel):
#     text: str

# def get_meta_features(text: str):
#     return {
#         "word_count": len(text.split()),
#         "char_count": len(text),
#         "exclamation_count": text.count('!'),
#         "question_count": text.count('?')
#     }

# def clean_input_text(text: str):
#     text = text.lower()
#     text = re.sub(r'<[^>]+>', ' ', text)
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#     text = re.sub(r'[^a-z\s]', '', text)
#     words = [w for w in text.split() if w not in stop_words]
#     return ' '.join(words)

# # --- 3. API Endpoints ---

# @app.get("/")
# def home():
#     return {"message": "Sentiment Analysis API is Running"}

# @app.post("/predict")
# def predict(request: SentimentRequest):
#     raw_text = request.text
    
#     # 1. Extract Meta Features
#     meta = get_meta_features(raw_text)
    
#     # 2. Clean Text
#     cleaned = clean_input_text(raw_text)
    
#     # 3. Create a DataFrame match the training format
#     import pandas as pd
#     input_df = pd.DataFrame([{
#         'clean_text': cleaned,
#         'word_count': meta['word_count'],
#         'char_count': meta['char_count'],
#         'exclamation_count': meta['exclamation_count'],
#         'question_count': meta['question_count']
#     }])
    
#     # 4. Predict
#     prediction = model.predict(input_df)[0]
#     label = "Positive" if prediction == 1 else "Negative"
    
#     # Get probability/confidence if using LinearSVC (requires extra steps)
#     # Since LinearSVC doesn't do predict_proba by default, we just return the label.
    
#     return {
#         "text": raw_text,
#         "sentiment": label,
#         "meta_features": meta
#     }


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os
import glob
import pandas as pd

# --- 1. Setup & Model Loading ---
app = FastAPI(title="Multi-Model Sentiment Analysis Engine")

# Enable CORS so the HTML frontend can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for local testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading all models from ./models/ directory...")
models = {}
model_files = glob.glob('./models/*.pkl')

if not model_files:
    print("WARNING: No .pkl files found in ./models/")

for file_path in model_files:
    # Extract just the filename without the .pkl extension (e.g., 'linear_imdb')
    model_name = os.path.basename(file_path).replace('.pkl', '')
    try:
        models[model_name] = joblib.load(file_path)
        print(f"Loaded: {model_name}")
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")

# --- 2. Preprocessing Logic ---
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
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# --- 3. API Endpoints ---

@app.get("/")
def home():
    return {"message": f"Engine Running. {len(models)} models loaded."}

@app.post("/predict")
def predict(request: SentimentRequest):
    raw_text = request.text
    
    # Extract and Clean
    meta = get_meta_features(raw_text)
    cleaned = clean_input_text(raw_text)
    
    # Create DataFrame
    input_df = pd.DataFrame([{
        'clean_text': cleaned,
        'word_count': meta['word_count'],
        'char_count': meta['char_count'],
        'exclamation_count': meta['exclamation_count'],
        'question_count': meta['question_count']
    }])
    
    # Loop through ALL loaded models and store their predictions
    predictions_results = {}
    for name, model in models.items():
        try:
            pred_value = model.predict(input_df)[0]
            predictions_results[name] = "Positive" if pred_value == 1 else "Negative"
        except Exception as e:
            predictions_results[name] = "Error"
    
    return {
        "text": raw_text,
        "clean_text_used": cleaned,
        "meta_features": meta,
        "predictions": predictions_results
    }