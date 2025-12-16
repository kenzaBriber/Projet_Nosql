# ml_gds_model.py
from __future__ import annotations

import os
import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


ART_PATH = os.path.join("ml_artifacts", "gds_multilabel.joblib")


def build_dataset_from_embeddings(ids, X, mongo_col, min_freq=20, label_field="go_terms"):
    """
    ids: list of protein ids (same key used in Mongo field 'entry')
    X: np.ndarray embeddings
    mongo_col: pymongo collection
    label_field: 'go_terms' or 'ec_numbers'
    """
    # Fetch labels from Mongo
    Y = []
    kept_ids = []
    kept_X = []

    for pid, emb in zip(ids, X):
        doc = mongo_col.find_one({"entry": pid}, {label_field: 1, "_id": 0})
        labels = (doc or {}).get(label_field, []) or []
        if labels:
            kept_ids.append(pid)
            kept_X.append(emb)
            Y.append(labels)

    if len(kept_ids) == 0:
        raise RuntimeError(f"Aucun label trouvé dans Mongo pour {label_field}.")

    kept_X = np.asarray(kept_X, dtype=np.float32)

    # Filter rare labels
    freq = {}
    for labels in Y:
        for lab in labels:
            freq[lab] = freq.get(lab, 0) + 1
    allowed = {lab for lab, c in freq.items() if c >= int(min_freq)}

    Y_filt = [[lab for lab in labels if lab in allowed] for labels in Y]
    mask = [len(labels) > 0 for labels in Y_filt]

    kept_ids = [i for i, m in zip(kept_ids, mask) if m]
    kept_X = kept_X[np.array(mask)]
    Y_filt = [labels for labels in Y_filt if labels]

    mlb = MultiLabelBinarizer()
    Y_bin = mlb.fit_transform(Y_filt)

    return kept_ids, kept_X, mlb, Y_bin


def train_ovr_logreg(X, Y_bin):
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=200)
    )
    clf.fit(X, Y_bin)
    return clf


def train_and_save_gds_models(ids, X, mongo_col, min_freq_go=20, min_freq_ec=10):
    

    # EC
    ec_ids, X_ec, ec_mlb, Y_ec = build_dataset_from_embeddings(
        ids, X, mongo_col, min_freq=min_freq_ec, label_field="ec_numbers"
    )
    ec_model = train_ovr_logreg(X_ec, Y_ec)

    # quick train metrics (micro F1 on train)
   
    ec_pred = ec_model.predict(X_ec)
  
    ec_f1 = float(f1_score(Y_ec, ec_pred, average="micro"))

    os.makedirs(os.path.dirname(ART_PATH), exist_ok=True)
    joblib.dump(
        {
            "ids": ids,              # ids embeddings (tous)
            "X": X,                  # embeddings (tous)
            "ec_model": ec_model,
            "ec_mlb": ec_mlb,
            "metrics": {"ec_micro_f1_train": ec_f1},
            "params": {"min_freq_ec": min_freq_ec},
        },
        ART_PATH,
    )

    return {"ec_micro_f1_train": ec_f1}


def load_artifacts():
    if not os.path.exists(ART_PATH):
        raise RuntimeError("Aucun modèle sauvegardé. Clique d'abord sur 'Entraîner et sauvegarder'.")
    return joblib.load(ART_PATH)


def predict_topk_from_embedding(emb: np.ndarray, model, mlb, k: int = 10):
    # predict_proba -> scores
    probs = model.predict_proba(emb.reshape(1, -1))[0]
    idx = np.argsort(-probs)[:k]
    labels = mlb.classes_[idx]
    return [(str(l), float(probs[i])) for i, l in zip(idx, labels)]
