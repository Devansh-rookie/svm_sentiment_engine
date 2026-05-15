import pandas as pd
import time
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ==========================================
# HYPERPARAMETERS
# ==========================================
KERNEL_DATA_PERCENTAGE = 0.6

def handle_edge_cases(df):
    df['clean_text'] = df['clean_text'].replace(r'^\s*$', np.nan, regex=True)
    return df.dropna(subset=['clean_text', 'sentiment']).copy()

def build_pipelines():
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000), 'clean_text'),
            ('num', StandardScaler(), ['word_count', 'char_count', 'exclamation_count', 'question_count'])
        ]
    )
    linear_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', LinearSVC(C=1.0, max_iter=2000, class_weight='balanced'))])
    kernel_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', SVC(kernel='rbf', C=1.0, class_weight='balanced'))])
    return linear_pipe, kernel_pipe

def evaluate_and_save(algo_name, train_domain, pipeline, X_train, y_train, test_sets, metrics_list):
    """Trains, tests on all domains, saves the model, and logs metrics."""
    print(f"\n>> Training {algo_name} on {train_domain}...")
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"   Training Time: {train_time:.2f} seconds")
    
    # Save the specific model
    filename = f"{algo_name}_{train_domain}".replace(" ", "").lower()
    joblib.dump(pipeline, f'./models/{filename}.pkl')
    
    # Test on all datasets
    for test_domain, (X_test, y_test) in test_sets.items():
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        print(f"   -> Accuracy on {test_domain}: {acc:.4f}")
        
        # Log to our master list
        metrics_list.append({
            "Algorithm": algo_name,
            "Trained_On": train_domain,
            "Tested_On": test_domain,
            "Accuracy": acc,
            "Training_Time_Sec": train_time
        })
        
    return pipeline

def main():
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./outputs', exist_ok=True)
    metrics_list = []

    print("Loading datasets...")
    df_imdb = handle_edge_cases(pd.read_csv('./data/imdb_processed.csv'))
    df_twit = handle_edge_cases(pd.read_csv('./data/twitter_processed.csv'))

    features = ['clean_text', 'word_count', 'char_count', 'exclamation_count', 'question_count']

    # Slicing
    X_imdb, y_imdb = df_imdb[features], df_imdb['sentiment']
    X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(X_imdb, y_imdb, test_size=0.2, random_state=42)

    X_twit, y_twit = df_twit[features], df_twit['sentiment']
    X_train_twit, X_test_twit, y_train_twit, y_test_twit = train_test_split(X_twit, y_twit, test_size=0.2, random_state=42)

    df_mixed = pd.concat([df_imdb, df_twit], ignore_index=True).sample(frac=1, random_state=42)
    X_mix, y_mix = df_mixed[features], df_mixed['sentiment']
    X_train_mix, X_test_mix, y_train_mix, y_test_mix = train_test_split(X_mix, y_mix, test_size=0.2, random_state=42)

    linear_pipe, kernel_pipe = build_pipelines()

    test_suite = {
        "IMDB": (X_test_imdb, y_test_imdb),
        "Twitter": (X_test_twit, y_test_twit),
        "Combined": (X_test_mix, y_test_mix)
    }

    # PHASE 1: LINEAR SVM
    print("\n--- PHASE 1: LINEAR SVM ---")
    evaluate_and_save("Linear", "IMDB", linear_pipe, X_train_imdb, y_train_imdb, test_suite, metrics_list)
    evaluate_and_save("Linear", "Twitter", linear_pipe, X_train_twit, y_train_twit, test_suite, metrics_list)
    evaluate_and_save("Linear", "Combined", linear_pipe, X_train_mix, y_train_mix, test_suite, metrics_list)

    # PHASE 2: KERNEL SVM
    print(f"\n--- PHASE 2: KERNEL SVM ({KERNEL_DATA_PERCENTAGE * 100}% Data) ---")
    n_imdb, n_twit, n_mix = int(len(X_train_imdb)*KERNEL_DATA_PERCENTAGE), int(len(X_train_twit)*KERNEL_DATA_PERCENTAGE), int(len(X_train_mix)*KERNEL_DATA_PERCENTAGE)
    
    evaluate_and_save("Kernel", "IMDB", kernel_pipe, X_train_imdb[:n_imdb], y_train_imdb[:n_imdb], test_suite, metrics_list)
    evaluate_and_save("Kernel", "Twitter", kernel_pipe, X_train_twit[:n_twit], y_train_twit[:n_twit], test_suite, metrics_list)
    evaluate_and_save("Kernel", "Combined", kernel_pipe, X_train_mix[:n_mix], y_train_mix[:n_mix], test_suite, metrics_list)

    # Export Metrics to CSV for Graphing
    pd.DataFrame(metrics_list).to_csv('./outputs/model_metrics.csv', index=False)
    print("\nAll 6 models saved to ./models/")
    print("Performance metrics saved to ./outputs/model_metrics.csv")

if __name__ == "__main__":
    main()