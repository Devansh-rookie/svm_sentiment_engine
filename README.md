# Cross-Domain Sentiment Analysis Engine: SVM Architecture

## 📌 Project Overview
This project implements a highly robust **Sentiment Analysis Engine** using Support Vector Machines (SVM). Instead of relying on a single sanitized dataset, this engine was designed to evaluate **Domain Shift** by training and testing across highly divergent text sources: formal movie reviews and informal social media text.

The objective was to satisfy advanced practical grade requirements by evaluating:
* **Algorithm Efficiency:** Linear SVM vs. Kernel SVM (RBF)
* **Cross-Domain Generalization:** IMDB (Formal) vs. Twitter (Informal)
* **Advanced Feature Engineering:** TF-IDF + Heuristic Meta-Features
* **Edge Case Handling:** Dealing with empty arrays, heavy slang, and extreme dimensionality.
* **Ensemble Deployment:** Serving 6 distinct models concurrently via a FastAPI backend and interactive frontend.

---

## 💾 Dataset Architecture
To prove cross-domain generalization, two fundamentally different datasets were used:
1. **IMDB 50K Movie Reviews:** Long-form, highly formal, vocabulary-rich.
2. **Sentiment140 (Twitter):** Short-form, sparse, heavy slang, and emoji-laden.
3. **Combined Dataset:** A randomized, balanced mixture of both domains to create the ultimate generalized model.

---

## ⚙️ The Pipeline & Feature Engineering
A `ColumnTransformer` pipeline was constructed to handle two distinct data streams simultaneously:

**1. Text Vectorization (The Core):**
*   **TF-IDF Vectorizer:** Max 5,000 features.
*   **N-Grams:** Range (1, 2) utilized to capture contextual bigrams (e.g., "not good").
*   **Cleaning:** Regex-based removal of HTML (`<br/>`), URLs, `@mentions`, and NLTK stopword filtering.

**2. Meta-Feature Engineering (The Edge):**
Extracted numerically *before* text cleaning and normalized via `StandardScaler`:
*   `word_count` & `char_count`
*   `exclamation_count` (!) & `question_count` (?)

---

## 📊 Cross-Domain Evaluation (The "Domain Shift" Matrix)
The most critical finding of this engine was observing how models collapse when exposed to unseen domains. 

### **Linear SVM Performance Matrix (F1-Score / Accuracy)**
| Trained On | Tested on IMDB | Tested on Twitter | Tested on Combined |
| :--- | :--- | :--- | :--- |
| **IMDB Only** | **0.887** / 88.6% | 0.615 / 58.2% *(Crash)* | 0.717 / 70.0% |
| **Twitter Only**| 0.660 / 66.6% *(Crash)*| **0.770** / 76.5% | 0.755 / 75.4% |
| **Combined** | **0.890** / 88.8% | **0.765** / 76.0% | **0.791** / 78.8% 🏆 |

**Key Insights:**
1. **The Domain Penalty:** A model trained purely on IMDB collapses on Twitter data (88.6% -> 58.2% Accuracy). The model expects formal syntax and fails to interpret short-form internet slang.
2. **The Combined Supremacy:** Training on the **Combined Dataset** resulted in the most robust model. It actually scored *higher* on IMDB testing (0.890 F1) than the model trained exclusively on IMDB (0.887 F1), proving that introducing high-variance noise (Twitter data) improved the model's overall generalization.

---

## 🧮 Algorithm Comparison: Linear vs. Kernel (RBF)
Two algorithms were tested to evaluate computational cost vs. accuracy on sparse text matrices.

### **Model 1: Linear SVM (The Winner)**
*   **Best Accuracy (Combined):** 78.85% (F1: 0.791)
*   **Training Time:** ~7.35 seconds (on 150k rows)
*   **Complexity:** $O(n)$

### **Model 2: Kernel SVM (RBF)**
*   **Best Accuracy (Combined):** 78.79% (F1: 0.792)
*   **Training Time:** ~521.62 seconds (Requires aggressive data subsetting to prevent thermal throttling).
*   **Complexity:** $O(n^2)$ to $O(n^3)$

### 🚨 The Kernel SVM Catastrophe (Twitter -> IMDB)
When the Kernel SVM was trained on Twitter and tested on IMDB, it suffered a catastrophic failure:
*   **Accuracy:** 49.69%
*   **Precision:** 0.7857
*   **Recall:** **0.0021** (F1-Score: 0.0043)
*   **Why this happened:** The RBF kernel attempted to map the highly sparse, short-form Twitter data into an infinite-dimensional space. When exposed to dense, 300-word IMDB paragraphs, the distance calculations collapsed, causing the model to predict nearly every single review as the majority negative class (hence the 0.002 recall). 

**Conclusion:** The **Linear SVM** vastly outperforms the Kernel SVM for NLP tasks. Text data vectorized via TF-IDF creates a highly dimensional space where data is already linearly separable. The RBF Kernel provides massive computational overhead with near-zero gain.

---

## 🚀 Interactive Ensemble Deployment
The project features a full-stack deployment architecture to evaluate all 6 trained models concurrently in real-time.

* **Backend:** FastAPI REST API (`/predict`). Dynamically loads all serialized `.pkl` models into memory and processes incoming requests through the shared `ColumnTransformer`.
* **Frontend:** A responsive HTML/TailwindCSS dashboard that extracts text, displays live meta-features (word counts, punctuation intensity), and outputs a comparative matrix of how the different domain models predict the exact same text.

---

## 🔮 Future Improvements & Deep Learning Caveat
While this TF-IDF + SVM baseline proved highly efficient and mathematically fascinating regarding domain shift, it has inherent algorithmic limitations:
1.  **Context Loss:** TF-IDF fundamentally ignores sequential word order outside of the defined (1, 2) n-grams.
2.  **Sarcasm:** SVMs cannot easily parse complex negation or semantic sarcasm.

**Next Steps:** As part of a Deep Learning curriculum, the next logical evolution of this engine is to replace the SVM with a neural architecture—such as an **LSTM (Long Short-Term Memory)** network or a **Transformer (BERT)**. These models natively capture sequential context and attention, resolving the sparsity issues that crashed the SVMs, though they require exponentially higher computational resources.
