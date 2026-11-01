import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1) Data (same file and split as train_model.py)
df = pd.read_csv("balanced_reviews.csv").dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"],
    test_size=0.2, random_state=42, stratify=df["sentiment"]
)

# 2) Three models, same TF-IDF features
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "SVM (LinearSVC)": LinearSVC(class_weight="balanced"),
}

results, trained = [], {}
for name, clf in classifiers.items():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
        ("classifier", clf),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro")
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "Precision": round(p * 100, 2),
        "Recall": round(r * 100, 2),
        "F1-score": round(f1 * 100, 2),
    })
    trained[name] = model

# 3) Comparison table
table = pd.DataFrame(results)
print("\nModel comparison (values in %):")
print(table.to_string(index=False))
table.to_csv("model_comparison.csv", index=False)

# 4) Save the best model (by F1) where app.py loads it from
best_name = table.sort_values("F1-score", ascending=False).iloc[0]["Model"]
os.makedirs("models", exist_ok=True)
joblib.dump(trained[best_name], "models/sentiment_model.pkl")
print("\nBest model:", best_name, "-> saved to models/sentiment_model.pkl")