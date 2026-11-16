"""Evaluation metrics for GridGuard (unstable = positive class)."""
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def evaluate(models, X, y):
    """Score fitted models on (X, y) and return one row of metrics per model."""
    rows = {}
    for name, model in models.items():
        pred = model.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows[name] = {
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        }
    return pd.DataFrame(rows).T