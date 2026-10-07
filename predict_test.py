"""
HASTIKA @ ICON-2026 - Final test-set prediction script
Trains on the full labeled training data and predicts on the official
(gold-label-withheld) test files released by the organizers on 30 Sep 2026.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import re

RANDOM_STATE = 42

def clean_text(s):
    s = str(s)
    s = re.sub(r'<br\s*/?>', ' ', s)
    s = re.sub(r'&quot;', '"', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def make_features():
    word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    return word_vec, char_vec

def fit_transform_both(word_vec, char_vec, texts):
    return hstack([word_vec.fit_transform(texts), char_vec.fit_transform(texts)]).tocsr()

def transform_both(word_vec, char_vec, texts):
    return hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()

def run(train_path, test_path, text_col, label_col, id_col, out_path, name):
    print(f"\n===== {name} =====")
    df = pd.read_csv(train_path)
    df[text_col] = df[text_col].map(clean_text)

    # quick internal validation check
    Xtr_t, Xdv_t, ytr, ydv = train_test_split(df[text_col], df[label_col], test_size=0.15,
                                               random_state=RANDOM_STATE, stratify=df[label_col])
    wv, cv = make_features()
    Xtr = fit_transform_both(wv, cv, Xtr_t)
    Xdv = transform_both(wv, cv, Xdv_t)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=5)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xdv)
    print(f"Internal Macro-F1: {f1_score(ydv, pred, average='macro'):.4f}  Acc: {accuracy_score(ydv, pred):.4f}")

    # retrain on full labeled data
    wv_f, cv_f = make_features()
    Xfull = fit_transform_both(wv_f, cv_f, df[text_col])
    clf_f = LogisticRegression(max_iter=2000, class_weight='balanced', C=5)
    clf_f.fit(Xfull, df[label_col])

    test_df = pd.read_csv(test_path)
    test_df[text_col] = test_df[text_col].map(clean_text)
    Xtest = transform_both(wv_f, cv_f, test_df[text_col])
    test_pred = clf_f.predict(Xtest)

    out = pd.DataFrame({id_col: test_df[id_col], 'label': test_pred})
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} ({len(out)} rows)")
    print(out['label'].value_counts())

if __name__ == "__main__":
    run(
        train_path="binary_train.csv",
        test_path="hastika_binary_test.csv",
        text_col="Comment", label_col="Label", id_col="id",
        out_path="JunaidTeam_taskA.csv",
        name="Task A - FINAL test predictions"
    )
    run(
        train_path="multiclass_train.csv",
        test_path="hastika_multiclass_test.csv",
        text_col="Comment", label_col="Hate Category", id_col="id",
        out_path="JunaidTeam_taskB.csv",
        name="Task B - FINAL test predictions"
    )
