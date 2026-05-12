# Sentiment Analysis Engine — Model Training Report

## Project Overview

This project implements a **Sentiment Analysis Engine** using **Support Vector Machines (SVMs)** for binary text classification on the IMDB movie reviews dataset.

Two different SVM models were trained and compared:

1. **Linear SVM**
2. **Kernel SVM (RBF Kernel)**

The objective was to evaluate:

* Accuracy
* Training speed
* Precision / Recall / F1-score
* Scalability for large text datasets

---

# Dataset Information

* **Dataset:** IMDB Movie Reviews
* **Task:** Binary Sentiment Classification
* **Classes:**

  * `0` → Negative Review
  * `1` → Positive Review

---

# Feature Engineering Pipeline

## Text Preprocessing

* Lowercasing
* Stopword removal
* Tokenization
* Noise cleaning

## Vectorization

TF-IDF Vectorization was used to convert text into numerical feature vectors.

Features included:

* Unigrams
* N-grams
* Sparse high-dimensional vectors

---

# Model 1 — Linear SVM

## Training Details

* **Algorithm:** Linear Support Vector Machine
* **Library:** LibLinear
* **Kernel:** Linear
* **Training Time:** **7.35 seconds**

---

## Training Logs

```text
iter 1 act 2.290e+04 pre 2.177e+04 delta 1.481e+01 f 4.000e+04 |g| 1.412e+04 CG 6
iter 2 act 4.860e+03 pre 4.197e+03 delta 1.784e+01 f 1.710e+04 |g| 2.240e+03 CG 8
iter 3 act 1.700e+03 pre 1.498e+03 delta 2.077e+01 f 1.224e+04 |g| 8.619e+02 CG 14
...
[LibLinear]
```

---

## Performance Metrics

| Metric        | Value        |
| ------------- | ------------ |
| Accuracy      | **88.56%**   |
| Precision     | 0.89         |
| Recall        | 0.89         |
| F1-Score      | 0.89         |
| Training Time | **7.35 sec** |

---

## Classification Report

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Negative (0) | 0.89      | 0.88   | 0.88     | 4961    |
| Positive (1) | 0.88      | 0.90   | 0.89     | 5039    |

---

# Model 2 — Kernel SVM (RBF)

## Training Details

* **Algorithm:** Support Vector Machine
* **Kernel:** Radial Basis Function (RBF)
* **Library:** LibSVM
* **Training Time:** **521.62 seconds**

---

## Training Logs

```text
optimization finished, #iter = 21958
obj = -13782.596013
rho = -0.268293
nSV = 20668
nBSV = 15636
Total nSV = 20668
[LibSVM]
```

---

## Performance Metrics

| Metric        | Value          |
| ------------- | -------------- |
| Accuracy      | **87.68%**     |
| Precision     | 0.88           |
| Recall        | 0.88           |
| F1-Score      | 0.88           |
| Training Time | **521.62 sec** |

---

## Classification Report

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Negative (0) | 0.88      | 0.87   | 0.87     | 4961    |
| Positive (1) | 0.87      | 0.89   | 0.88     | 5039    |

---

# Comparative Analysis

| Feature       | Linear SVM             | Kernel SVM (RBF)        |
| ------------- | ---------------------- | ----------------------- |
| Accuracy      | **88.56%**             | 87.68%                  |
| Training Time | **7.35 sec**           | 521.62 sec              |
| Kernel Type   | Linear                 | RBF                     |
| Scalability   | Excellent              | Poor for large datasets |
| Complexity    | Lower                  | Higher                  |
| Memory Usage  | Lower                  | Higher                  |
| Best Use Case | Large sparse text data | Complex nonlinear data  |

---

# Key Observations

## 1. Linear SVM Performed Better

The Linear SVM achieved:

* Higher accuracy
* Faster convergence
* Better scalability

This is expected because TF-IDF text data is:

* High-dimensional
* Sparse
* Often linearly separable

---

## 2. RBF Kernel Was Computationally Expensive

The RBF kernel:

* Took approximately **71× more training time**
* Used a very large number of support vectors
* Did not improve accuracy significantly

---

## 3. Why Linear SVM Works Well for NLP

Linear SVMs are highly effective for:

* Text classification
* Spam detection
* Sentiment analysis
* Document categorization

Because:

* Text vectors are sparse
* Dimensionality is extremely high
* Linear boundaries often generalize well

---

# Final Conclusion

The experiment demonstrates that **Linear SVM is the better choice for sentiment analysis on large text datasets**.

## Final Recommendation

✅ Use **Linear SVM** for:

* Faster training
* Better scalability
* Comparable or better accuracy
* Production deployment

❌ Avoid RBF Kernel SVM for very large NLP datasets because:

* Training becomes extremely slow
* Memory consumption increases significantly
* Accuracy improvement is negligible

---

# Saved Models

```text
models/
├── linear_svm_model.pkl
└── kernel_svm_model.pkl
```

---

# Technologies Used

* Python
* Scikit-learn
* TF-IDF Vectorizer
* NumPy
* Pandas
* LibLinear
* LibSVM

---

# Future Improvements

Possible enhancements:

* Hyperparameter tuning using GridSearchCV
* Using Word Embeddings (Word2Vec / GloVe)
* Deep Learning models:

  * LSTM
  * GRU
  * Transformers (BERT)
* Real-time sentiment prediction API deployment
* Web interface using Streamlit or Flask
