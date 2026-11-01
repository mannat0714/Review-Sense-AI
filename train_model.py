import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Dataset padho
df = pd.read_csv("balanced_reviews.csv")
df = df.dropna()

print("Total reviews:", len(df))

# 80% training ke liye, 20% testing ke liye
X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# Model banao
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Model train karo
model.fit(X_train, y_train)

# Model ko test karo
predictions = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nDetailed report:")
print(classification_report(y_test, predictions))

# Model save karo
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.pkl")

print("Model training completed!")
print("Model saved successfully!")

# Sample reviews test karo
test_reviews = [
    "This product is amazing",
    "This product is terrible",
    "This product is okay okay",
    "The product is not good",
    "The product is good but expensive"
]

print("\nSample predictions:")
for review in test_reviews:
    prediction = model.predict([review])[0]
    print(review, "-->", prediction)