from flask import Flask, render_template, request, session, redirect, url_for, send_file
import joblib
import os
import re
import pandas as pd
from product_analysis import register_product_routes
from model_insights import top_words, load_comparison
from hybrid import HybridModel, decide_single

app = Flask(__name__)
app.secret_key = "reviewsense_demo_key"

model = joblib.load("models/sentiment_model.pkl")
hybrid = HybridModel(model)          # ML model + common-sense word rules
register_product_routes(app, hybrid)

# ---------- Buying recommendation shown under the result ----------
RECOMMENDATIONS = {
    "Positive": {"icon": "\u2705", "text": "This is worth it to buy!", "bg": "#dcfce7", "color": "#166534"},
    "Negative": {"icon": "\u274c", "text": "Not recommended to buy.", "bg": "#fee2e2", "color": "#991b1b"},
    "Neutral": {"icon": "\U0001F914", "text": "Average product, think before buying.", "bg": "#e0e7ff", "color": "#3730a3"},
}

# ---------- Tricky / mixed sentences ("amazing but not worth it") ----------
CONTRAST_PATTERN = re.compile(r"\b(?:but|however|nevertheless|nonetheless)\b", re.IGNORECASE)


def analyze_review(text):
    """Return (final_sentiment, clauses, is_mixed, confidence, method, evidence).

    If the review has a contrast word (but / however ...), each part is
    classified separately. The part after the contrast word usually carries
    the customer's real opinion, so the last part decides the final result.
    Every part goes through the hybrid rule (ML model + common-sense words).
    """
    parts = [p.strip(" ,;.!-") for p in CONTRAST_PATTERN.split(text)]
    parts = [p for p in parts if p]

    if len(parts) < 2:
        label, conf, method, evidence = decide_single(model, text)
        return label, [], False, conf, method, evidence

    results = [decide_single(model, p) for p in parts]
    labels = [r[0] for r in results]
    clauses = [{"text": p, "sentiment": l} for p, l in zip(parts, labels)]
    is_mixed = len(set(labels)) > 1
    last = results[-1]
    return last[0], clauses, is_mixed, last[1], last[2], last[3]


LABELS = ["Positive", "Negative", "Neutral"]
MAX_BATCH_ROWS = 5000
RESULT_FILE = os.path.join(app.root_path, "batch_results.csv")
COMPARISON_FILE = os.path.join(app.root_path, "model_comparison.csv")
REVIEW_COLUMN_NAMES = [
    "review", "reviews.text", "reviews_text", "review_text",
    "text", "comment", "comments", "content",
]


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    review = ""
    clauses = []
    is_mixed = False
    recommendation = None
    confidence = None
    method = "model"
    evidence = []

    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if review:
            sentiment, clauses, is_mixed, confidence, method, evidence = analyze_review(review)
            recommendation = RECOMMENDATIONS.get(str(sentiment))

            history = session["history"]
            history.insert(0, {
                "review": review,
                "sentiment": str(sentiment)
            })

            session["history"] = history[:10]

    history = session.get("history", [])

    counts = {
        "Positive": sum(1 for item in history if item["sentiment"] == "Positive"),
        "Negative": sum(1 for item in history if item["sentiment"] == "Negative"),
        "Neutral": sum(1 for item in history if item["sentiment"] == "Neutral")
    }

    return render_template(
        "index.html",
        sentiment=sentiment,
        review=review,
        history=history,
        counts=counts,
        clauses=clauses,
        is_mixed=is_mixed,
        recommendation=recommendation,
        confidence=confidence,
        method=method,
        evidence=evidence
    )


@app.route("/clear-history", methods=["POST"])
def clear_history():
    session["history"] = []
    return redirect(url_for("home"))


# ---------- Model insights: comparison chart + word clouds ----------

@app.route("/insights")
def insights():
    return render_template(
        "insights.html",
        comparison=load_comparison(COMPARISON_FILE),
        words=top_words(model, 18),
    )


# ---------- Batch analysis (CSV upload) ----------

def find_review_column(df):
    """Pick the column that holds the review text."""
    lowered = {str(c).lower().strip(): c for c in df.columns}
    for name in REVIEW_COLUMN_NAMES:
        if name in lowered:
            return lowered[name]
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        return None
    # fall back to the text column with the longest average length
    return max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())


@app.route("/batch", methods=["GET", "POST"])
def batch():
    error = None
    summary = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            error = "Please choose a CSV file first."
        elif not file.filename.lower().endswith(".csv"):
            error = "Only .csv files are supported."
        else:
            try:
                df = pd.read_csv(file, encoding_errors="replace")
            except Exception:
                df = None
                error = "Could not read this file. Make sure it is a valid CSV."

            if df is not None:
                column = find_review_column(df)
                if column is None:
                    error = "No review text column found in this CSV."
                else:
                    reviews = df[column].dropna().astype(str).str.strip()
                    reviews = reviews[reviews.str.len() > 0]
                    all_count = len(reviews)
                    reviews = reviews.head(MAX_BATCH_ROWS)

                    if reviews.empty:
                        error = "The review column is empty."
                    else:
                        predictions = hybrid.predict(reviews.tolist())
                        out = pd.DataFrame({
                            "review": reviews.tolist(),
                            "sentiment": [str(p) for p in predictions],
                        })
                        out.to_csv(RESULT_FILE, index=False)

                        total = len(out)
                        counts = {l: int((out["sentiment"] == l).sum()) for l in LABELS}
                        percents = {l: round(counts[l] * 100 / total, 1) for l in LABELS}
                        summary = {
                            "column": column,
                            "total": total,
                            "truncated": all_count > MAX_BATCH_ROWS,
                            "limit": MAX_BATCH_ROWS,
                            "counts": counts,
                            "percents": percents,
                            "rows": out.head(20).to_dict("records"),
                        }

    return render_template("batch.html", error=error, summary=summary)


@app.route("/batch/download")
def batch_download():
    if not os.path.exists(RESULT_FILE):
        return redirect(url_for("batch"))
    return send_file(RESULT_FILE, as_attachment=True, download_name="batch_results.csv")


if __name__ == "__main__":
    app.run(debug=True) 