import pandas as pd
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    print(f"\n--- Starting Full Training: {name} ---")
    print("Note: If using RBF Kernel, this may take several minutes. Check terminal logs below.")
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    
    predictions = pipeline.predict(X_test)
    
    print(f"\n{name} Training Complete!")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Classification Report:\n{classification_report(y_test, predictions)}")
    return pipeline

def main():
    # Ensure models directory exists
    os.makedirs('./models', exist_ok=True)

    print("Loading processed IMDB data...")
    df = pd.read_csv('./data/imdb_processed.csv').dropna()
    
    X = df[['clean_text', 'word_count', 'char_count', 'exclamation_count', 'question_count']]
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Setup the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000), 'clean_text'),
            ('num', StandardScaler(), ['word_count', 'char_count', 'exclamation_count', 'question_count'])
        ]
    )

    # --- 1. Linear SVM (The Fast One) ---
    # verbose=1 shows the optimization iterations
    linear_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LinearSVC(C=1.0, max_iter=2000, verbose=1))
    ])
    
    # --- 2. Kernel SVM (The Heavy One) ---
    # verbose=True shows progress in the underlying C++ libsvm
    kernel_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', C=1.0, verbose=True))
    ])

    # Execute Full Training for Linear
    train_and_evaluate("Linear SVM", linear_pipe, X_train, y_train, X_test, y_test)
    
    # Execute Full Training for Kernel (Warning: This will take time!)
    # I removed the [:10000] slice so it uses all 40,000 training rows.
    train_and_evaluate("Kernel SVM (RBF)", kernel_pipe, X_train, y_train, X_test, y_test)

    # 4. Save BOTH models
    joblib.dump(linear_pipe, './models/linear_svm_model.pkl')
    joblib.dump(kernel_pipe, './models/kernel_svm_model.pkl')
    
    print("\nSuccess! Both models saved in ./models/")
    print("- linear_svm_model.pkl")
    print("- kernel_svm_model.pkl")

if __name__ == "__main__":
    main()