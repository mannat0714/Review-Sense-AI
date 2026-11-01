import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Same data and same split as train_model.py / compare_models.py, so numbers are comparable
df = pd.read_csv("balanced_reviews.csv").dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"],
    test_size=0.2, random_state=42, stratify=df["sentiment"]
)


def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)


candidates = {
    "1. Baseline: word TF-IDF + SVM": (
        Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
                  ("clf", LinearSVC(class_weight="balanced"))]),
        {"clf__C": [1.0]},
    ),
    "2. Word + character TF-IDF + SVM (tuned C)": (
        Pipeline([("feats", FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])),
                  ("clf", LinearSVC(class_weight="balanced"))]),
        {"clf__C": [0.05, 0.1, 0.3, 1.0]},
    ),
    "3. Word + character TF-IDF + Logistic Regression (tuned C)": (
        Pipeline([("feats", FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        {"clf__C": [0.5, 1.0, 3.0, 10.0]},
    ),
}

# Optional hybrid: TF-IDF + VADER lexicon scores (needs: pip install vaderSentiment)
try:
    from custom_features import VaderFeatures
    candidates["4. Hybrid: TF-IDF + VADER scores + SVM (tuned C)"] = (
        Pipeline([("feats", FeatureUnion([("word", word_tfidf()), ("char", char_tfidf()),
                                          ("vader", VaderFeatures())])),
                  ("clf", LinearSVC(class_weight="balanced"))]),
        {"clf__C": [0.05, 0.1, 0.3, 1.0]},
    )
except ImportError:
    print("vaderSentiment not installed - skipping the hybrid model "
          "(install with: pip install vaderSentiment)\n")

results, fitted = [], {}
for name, (pipe, grid) in candidates.items():
    print("Training:", name)
    # C is chosen with cross-validation on the TRAINING data only
    search = GridSearchCV(pipe, grid, cv=3, scoring="f1_macro", n_jobs=1)
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro")
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "Precision": round(p * 100, 2),
        "Recall": round(r * 100, 2),
        "F1-score": round(f1 * 100, 2),
    })
    fitted[name] = search.best_estimator_

table = pd.DataFrame(results)
print("\nResults (values in %):")
print(table.to_string(index=False))
table.to_csv("improved_model_comparison.csv", index=False)

best_name = table.sort_values("F1-score", ascending=False).iloc[0]["Model"]
print("\nBest:", best_name)

answer = input("Save this as models/sentiment_model.pkl for the website? (y/n): ").strip().lower()
if answer == "y":
    os.makedirs("models", exist_ok=True)
    joblib.dump(fitted[best_name], "models/sentiment_model.pkl")
    print("Saved. Restart the Flask app (python app.py) to use it.")
else:
    print("Not saved. The website keeps using the old model.")
