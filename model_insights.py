"""Helpers for: confidence score, top words (word cloud) and model comparison."""
import os
import random

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- confidence
def confidence_for(model, text):
    """Confidence (0-100) of the model's prediction for one text.

    LinearSVC has no probabilities, so we turn its decision scores into a
    softmax-style relative confidence. It is a relative score, not a
    calibrated probability.
    """
    try:
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba([text]))[0]
            return round(float(p.max()) * 100, 1)
        if hasattr(model, "decision_function"):
            s = np.asarray(model.decision_function([text]), dtype=float)
            if s.ndim == 2 and s.shape[1] > 1:
                z = s[0] * 3.0
                e = np.exp(z - z.max())
                return round(float(e.max() / e.sum()) * 100, 1)
            val = abs(float(s.ravel()[0]))
            return round(100.0 / (1.0 + np.exp(-3.0 * val)), 1)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- top words
def top_words(model, n=18):
    """Return {label: [{word, size, op}, ...]} of the words that push the
    model most strongly towards each class (from the classifier weights)."""
    try:
        vec = model.steps[0][1]
        clf = model.steps[-1][1]
        names = np.asarray(vec.get_feature_names_out())
        classes = [str(c) for c in clf.classes_]

        if hasattr(clf, "coef_"):
            coef = np.asarray(clf.coef_)
            if coef.shape[0] == 1 and len(classes) == 2:      # binary case
                coef = np.vstack([-coef[0], coef[0]])
        elif hasattr(clf, "feature_log_prob_"):               # Naive Bayes
            flp = np.asarray(clf.feature_log_prob_)
            coef = flp - flp.mean(axis=0)
        else:
            return {}
    except Exception:
        return {}

    rng = random.Random(7)
    out = {}
    for label in ("Positive", "Neutral", "Negative"):
        if label not in classes:
            continue
        row = coef[classes.index(label)]
        idx = np.argsort(row)[::-1][:n]
        weights = row[idx]
        lo, hi = float(weights.min()), float(weights.max())
        span = (hi - lo) or 1.0
        items = []
        for i, w in zip(idx, weights):
            norm = (float(w) - lo) / span
            items.append({"word": str(names[i]),
                          "size": round(0.9 + 1.3 * norm, 2),
                          "op": round(0.6 + 0.4 * norm, 2)})
        rng.shuffle(items)          # looks like a cloud instead of a sorted list
        out[label] = items
    return out


# ---------------------------------------------------------- model comparison
def _num(v):
    try:
        return float(str(v).replace("%", "").strip())
    except ValueError:
        return None


def load_comparison(path):
    """Read model_comparison.csv -> list of rows (values in %), or None."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    cols = {str(c).lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_model = pick("model", "algorithm", "classifier") or df.columns[0]
    metrics = {
        "accuracy": pick("accuracy", "acc"),
        "precision": pick("precision"),
        "recall": pick("recall"),
        "f1": pick("f1", "f1-score", "f1_score", "f1 score"),
    }

    rows = []
    for _, r in df.iterrows():
        row = {"model": str(r[c_model])}
        for key, col in metrics.items():
            row[key] = _num(r[col]) if col is not None else None
        rows.append(row)

    vals = [row[k] for row in rows for k in metrics if row[k] is not None]
    if vals and max(vals) <= 1.0:                    # fractions -> percent
        for row in rows:
            for k in metrics:
                if row[k] is not None:
                    row[k] *= 100
    for row in rows:
        for k in metrics:
            if row[k] is not None:
                row[k] = round(row[k], 2)

    best = max(rows, key=lambda r: r["accuracy"] or 0)["model"]
    for row in rows:
        row["best"] = row["model"] == best
    return rows