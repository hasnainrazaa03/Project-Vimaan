# vimaan_benchmark.py

import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer

# -------------------------
# Load Dataset
# -------------------------
data_file = "./ml_model/aviation_cmds.jsonl"
data = [json.loads(l) for l in open(data_file)]
texts = [d["text"] for d in data]
labels = [d["intent"] for d in data]

# Map labels to indices
label2idx = {l: i for i, l in enumerate(sorted(set(labels)))}
idx2label = {i: l for l, i in label2idx.items()}
y = np.array([label2idx[l] for l in labels])

# Train/Val/Test split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
)  # 0.125 * 0.8 ≈ 0.1

# -------------------------
# 1. Semi-External Model (TF-IDF + Logistic Regression)
# -------------------------
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)
Xtr_tfidf = tfidf.fit_transform(X_train)
Xte_tfidf = tfidf.transform(X_test)

clf_tfidf = LogisticRegression(max_iter=1000)
clf_tfidf.fit(Xtr_tfidf, y_train)
pred_tfidf = clf_tfidf.predict(Xte_tfidf)

acc_tfidf = accuracy_score(y_test, pred_tfidf)
f1_tfidf = f1_score(y_test, pred_tfidf, average="macro")
print("Semi-External (TF-IDF + LR) Accuracy:", acc_tfidf)
print("Macro F1:", f1_tfidf)
print(classification_report(y_test, pred_tfidf, target_names=idx2label.values()))

# -------------------------
# 2. Fully External Model (Sentence Embeddings + Logistic Regression)
# -------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
Xtr_emb = embed_model.encode(X_train, show_progress_bar=True)
Xte_emb = embed_model.encode(X_test, show_progress_bar=True)

clf_emb = LogisticRegression(max_iter=1000)
clf_emb.fit(Xtr_emb, y_train)
pred_emb = clf_emb.predict(Xte_emb)

acc_emb = accuracy_score(y_test, pred_emb)
f1_emb = f1_score(y_test, pred_emb, average="macro")
print("Fully External (Embeddings + LR) Accuracy:", acc_emb)
print("Macro F1:", f1_emb)
print(classification_report(y_test, pred_emb, target_names=idx2label.values()))

# -------------------------
# 3. From-Scratch Model (BoW + NumPy Logistic Regression)
# -------------------------
# Build vocabulary
vocab = list(set(word for t in X_train for word in t.lower().split()))
word2idx = {w: i for i, w in enumerate(vocab)}
num_classes = len(label2idx)

def encode(text):
    vec = np.zeros(len(vocab))
    for word in text.lower().split():
        if word in word2idx:
            vec[word2idx[word]] = 1
    return vec

Xtr_bow = np.array([encode(t) for t in X_train])
Xte_bow = np.array([encode(t) for t in X_test])

# One-hot labels
Ytr = np.zeros((len(y_train), num_classes))
Ytr[np.arange(len(y_train)), y_train] = 1

# Initialize weights
W = np.random.randn(len(vocab), num_classes) * 0.01
b = np.zeros((1, num_classes))
lr = 0.1
epochs = 500

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Training loop
for epoch in range(epochs):
    z = Xtr_bow.dot(W) + b
    probs = softmax(z)
    grad = (probs - Ytr) / Xtr_bow.shape[0]
    W -= lr * Xtr_bow.T.dot(grad)
    b -= lr * np.sum(grad, axis=0, keepdims=True)
    if epoch % 100 == 0:
        loss = -np.mean(np.sum(Ytr * np.log(probs + 1e-9), axis=1))
        print(f"Epoch {epoch}, Loss={loss:.4f}")

# Evaluation
probs_test = softmax(Xte_bow.dot(W) + b)
pred_bow = np.argmax(probs_test, axis=1)
acc_bow = accuracy_score(y_test, pred_bow)
f1_bow = f1_score(y_test, pred_bow, average="macro")
print("From-Scratch (BoW + NumPy LR) Accuracy:", acc_bow)
print("Macro F1:", f1_bow)
