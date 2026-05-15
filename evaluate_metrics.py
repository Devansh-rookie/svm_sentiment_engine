import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==========================================
# HELPERS
# ==========================================

def handle_edge_cases(df):
    df['clean_text'] = df['clean_text'].replace(
        r'^\s*$',
        np.nan,
        regex=True
    )
    return df.dropna(
        subset=['clean_text', 'sentiment']
    ).copy()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1_Score": f1_score(y_test, y_pred),
        "Macro_F1": f1_score(
            y_test,
            y_pred,
            average='macro'
        ),
        "Classification_Report":
            classification_report(
                y_test,
                y_pred
            )
    }


def main():
    print("Loading datasets...")

    df_imdb = handle_edge_cases(
        pd.read_csv('./data/imdb_processed.csv')
    )

    df_twit = handle_edge_cases(
        pd.read_csv('./data/twitter_processed.csv')
    )

    features = [
        'clean_text',
        'word_count',
        'char_count',
        'exclamation_count',
        'question_count'
    ]

    # SAME SPLITS AS TRAINING
    X_imdb = df_imdb[features]
    y_imdb = df_imdb['sentiment']

    X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = (
        train_test_split(
            X_imdb,
            y_imdb,
            test_size=0.2,
            random_state=42
        )
    )

    X_twit = df_twit[features]
    y_twit = df_twit['sentiment']

    X_train_twit, X_test_twit, y_train_twit, y_test_twit = (
        train_test_split(
            X_twit,
            y_twit,
            test_size=0.2,
            random_state=42
        )
    )

    df_mix = pd.concat(
        [df_imdb, df_twit],
        ignore_index=True
    ).sample(frac=1, random_state=42)

    X_mix = df_mix[features]
    y_mix = df_mix['sentiment']

    X_train_mix, X_test_mix, y_train_mix, y_test_mix = (
        train_test_split(
            X_mix,
            y_mix,
            test_size=0.2,
            random_state=42
        )
    )

    test_suite = {
        "IMDB": (X_test_imdb, y_test_imdb),
        "Twitter": (X_test_twit, y_test_twit),
        "Combined": (X_test_mix, y_test_mix)
    }

    models = {
        "Linear_IMDB":
            "./models/linear_imdb.pkl",

        "Linear_Twitter":
            "./models/linear_twitter.pkl",

        "Linear_Combined":
            "./models/linear_combined.pkl",

        "Kernel_IMDB":
            "./models/kernel_imdb.pkl",

        "Kernel_Twitter":
            "./models/kernel_twitter.pkl",

        "Kernel_Combined":
            "./models/kernel_combined.pkl",
    }

    results = []

    for model_name, model_path in models.items():

        print(f"\nLoading {model_name}")

        model = joblib.load(model_path)

        for test_name, (X_test, y_test) in test_suite.items():

            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            print(
                f"{model_name} -> "
                f"{test_name}"
            )

            print(
                f"Accuracy: "
                f"{metrics['Accuracy']:.4f}"
            )

            print(
                f"F1: "
                f"{metrics['F1_Score']:.4f}"
            )

            results.append({
                "Model": model_name,
                "Tested_On": test_name,
                "Accuracy":
                    metrics["Accuracy"],
                "Precision":
                    metrics["Precision"],
                "Recall":
                    metrics["Recall"],
                "F1_Score":
                    metrics["F1_Score"],
                "Macro_F1":
                    metrics["Macro_F1"]
            })

    os.makedirs("./outputs", exist_ok=True)

    pd.DataFrame(results).to_csv(
        "./outputs/full_metrics.csv",
        index=False
    )

    print(
        "\nSaved metrics to "
        "./outputs/full_metrics.csv"
    )


if __name__ == "__main__":
    main()