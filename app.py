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
import numpy as np
import sklearn.compose._column_transformer

# Monkey-patch for scikit-learn < 1.5 or > 1.7 compatibility when unpickling 1.5 models
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        def __repr__(self):
            return "remainder"
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Peak evaluation accuracies achieved by each model across all benchmark domains (IMDB, Twitter, Combined)
MODEL_ACCURACIES = {
    "linear_imdb": 0.8856,
    "linear_twitter": 0.7656,
    "linear_combined": 0.8888,
    "kernel_imdb": 0.8682,
    "kernel_twitter": 0.7566,
    "kernel_combined": 0.8850
}

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
    total_weight = 0.0
    weighted_prob_sum = 0.0
    
    for name, model in models.items():
        try:
            pred_value = model.predict(input_df)[0]
            label = "Positive" if pred_value == 1 else "Negative"
            
            # Convert decision function distance to probability/confidence
            prob_pos = 0.5
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(input_df)[0]
                    prob_pos = probs[1]
                except Exception:
                    pass
            
            if prob_pos == 0.5 and hasattr(model, "decision_function"):
                dist = model.decision_function(input_df)[0]
                prob_pos = float(1.0 / (1.0 + np.exp(-dist)))
            
            conf_val = prob_pos if label == "Positive" else (1.0 - prob_pos)
            conf_str = f"{conf_val * 100:.1f}%"
            
            weight = MODEL_ACCURACIES.get(name, 0.75)
            total_weight += weight
            weighted_prob_sum += (weight * prob_pos)
            
            predictions_results[name] = {
                "sentiment": label,
                "confidence": conf_str,
                "confidence_val": float(conf_val),
                "prob_pos": float(prob_pos),
                "model_accuracy": weight
            }
        except Exception as e:
            predictions_results[name] = {
                "sentiment": "Error",
                "confidence": "0%",
                "confidence_val": 0.0,
                "prob_pos": 0.5,
                "model_accuracy": 0.0
            }
            
    final_sentiment = "Neutral"
    final_conf_str = "0%"
    if total_weight > 0:
        final_prob_pos = weighted_prob_sum / total_weight
        final_sentiment = "Positive" if final_prob_pos >= 0.5 else "Negative"
        final_conf_val = final_prob_pos if final_sentiment == "Positive" else (1.0 - final_prob_pos)
        final_conf_str = f"{final_conf_val * 100:.1f}%"
        
    return {
        "text": raw_text,
        "clean_text_used": cleaned,
        "meta_features": meta,
        "predictions": predictions_results,
        "ensemble_consensus": {
            "sentiment": final_sentiment,
            "confidence": final_conf_str
        }
    }