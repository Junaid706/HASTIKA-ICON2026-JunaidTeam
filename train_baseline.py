"""
HASTIKA @ ICON-2026 - Baseline pipeline
Task A: Binary Hate Speech Detection (Hate / Non-Hate)
Task B: Fine-Grained Hate Speech Classification (6 categories)

Approach: TF-IDF (word + char n-grams) + Logistic Regression.
Char n-grams help a lot with code-mixed / transliterated (Kanglish) text
because spelling varies a lot from comment to comment.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack
import re
import json

RANDOM_STATE = 42

def clean_text(s):
    s = str(s)
    s = re.sub(r'<br\s*/?>', ' ', s)          # stray html tags seen in the data
    s = re.sub(r'&quot;', '"', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'http\S+|www\.\S+', ' ', s)    # urls
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def make_features():
    """word-level + char-level TF-IDF, combined."""
    word_vec = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True
    )
    return word_vec, char_vec


def fit_transform_both(word_vec, char_vec, texts):
    Xw = word_vec.fit_transform(texts)
    Xc = char_vec.fit_transform(texts)
    return hstack([Xw, Xc]).tocsr()


def transform_both(word_vec, char_vec, texts):
    Xw = word_vec.transform(texts)
    Xc = char_vec.transform(texts)
    return hstack([Xw, Xc]).tocsr()


def run_task(train_path, val_inputs_path, text_col, label_col,
             id_col, out_pred_col, submission_path, report_name):
    print(f"\n===== {report_name} =====")
    df = pd.read_csv(train_path)
    df[text_col] = df[text_col].map(clean_text)

    # internal train/validation split (from labeled train data) to measure macro-F1
    X_train_text, X_dev_text, y_train, y_dev = train_test_split(
        df[text_col], df[label_col], test_size=0.15,
        random_state=RANDOM_STATE, stratify=df[label_col]
    )

    word_vec, char_vec = make_features()
    X_train = fit_transform_both(word_vec, char_vec, X_train_text)
    X_dev = transform_both(word_vec, char_vec, X_dev_text)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=5)
    clf.fit(X_train, y_train)

    dev_pred = clf.predict(X_dev)
    macro_f1 = f1_score(y_dev, dev_pred, average='macro')
    acc = accuracy_score(y_dev, dev_pred)
    print(f"Internal held-out Macro-F1: {macro_f1:.4f}")
    print(f"Internal held-out Accuracy: {acc:.4f}")
    print(classification_report(y_dev, dev_pred))

    # ---- retrain on FULL labeled data, then predict on the real validation_inputs file ----
    word_vec_f, char_vec_f = make_features()
    X_full = fit_transform_both(word_vec_f, char_vec_f, df[text_col])
    clf_full = LogisticRegression(max_iter=2000, class_weight='balanced', C=5)
    clf_full.fit(X_full, df[label_col])

    val_df = pd.read_csv(val_inputs_path)
    val_df[text_col] = val_df[text_col].map(clean_text)
    X_val = transform_both(word_vec_f, char_vec_f, val_df[text_col])
    val_pred = clf_full.predict(X_val)

    submission = pd.DataFrame({
        id_col: val_df[id_col],
        out_pred_col: val_pred
    })
    submission.to_csv(submission_path, index=False)
    print(f"Saved predictions -> {submission_path}  ({len(submission)} rows)")

    return {
        "task": report_name,
        "internal_macro_f1": macro_f1,
        "internal_accuracy": acc,
        "n_train": len(df),
        "n_val_predicted": len(submission)
    }


if __name__ == "__main__":
    results = []

    results.append(run_task(
        train_path="/mnt/user-data/uploads/binary_train.csv",
        val_inputs_path="/mnt/user-data/uploads/binary_validation_inputs.csv",
        text_col="Comment",
        label_col="Label",
        id_col="id",
        out_pred_col="Label",
        submission_path="/home/claude/hastika/binary_predictions.csv",
        report_name="Task A - Binary Hate Speech Detection"
    ))

    results.append(run_task(
        train_path="/mnt/user-data/uploads/multiclass_train.csv",
        val_inputs_path="/mnt/user-data/uploads/multiclass_validation_inputs.csv",
        text_col="Comment",
        label_col="Hate Category",
        id_col="id",
        out_pred_col="Hate Category",
        submission_path="/home/claude/hastika/multiclass_predictions.csv",
        report_name="Task B - Fine-Grained Hate Speech Classification"
    ))

    with open("/home/claude/hastika/results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\nSummary:")
    for r in results:
        print(r)
