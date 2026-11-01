import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderFeatures(BaseEstimator, TransformerMixin):
    """Turns each review into 4 numbers: VADER neg, neu, pos and compound score."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        analyzer = SentimentIntensityAnalyzer()
        rows = []
        for text in X:
            s = analyzer.polarity_scores(str(text))
            rows.append([s["neg"], s["neu"], s["pos"], s["compound"]])
        return np.array(rows)
