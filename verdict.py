"""Sentiment analysis of many reviews + buy / don't buy verdict."""
from collections import Counter

MAX_REVIEWS = 5000


def analyse(model, reviews):
    """reviews: list of {"text":..., "rating":...}. Returns summary dict."""
    reviews = [r for r in reviews if r.get("text") and str(r["text"]).strip()][:MAX_REVIEWS]
    if not reviews:
        return None

    texts = [str(r["text"]) for r in reviews]
    preds = [str(p) for p in model.predict(texts)]
    total = len(preds)
    counts = Counter(preds)

    pct = {k: round(counts.get(k, 0) * 100 / total, 1) for k in ("Positive", "Neutral", "Negative")}
    cnt = {k: counts.get(k, 0) for k in ("Positive", "Neutral", "Negative")}

    ratings = []
    for r in reviews:
        try:
            if r.get("rating") is not None:
                ratings.append(float(r["rating"]))
        except (TypeError, ValueError):
            pass
    avg = round(sum(ratings) / len(ratings), 2) if ratings else None

    samples = {"Positive": [], "Neutral": [], "Negative": []}
    for text, label in zip(texts, preds):
        if label in samples and len(samples[label]) < 3:
            samples[label].append(text[:300] + ("..." if len(text) > 300 else ""))

    return {"total": total, "pct": pct, "counts": cnt, "avg_rating": avg, "samples": samples}


def buy_verdict(pct, avg_rating, total):
    """Heuristic rules (thresholds chosen by us, tune after testing)."""
    pos, neg = pct["Positive"], pct["Negative"]
    avg = avg_rating if avg_rating is not None else None

    if pos >= 60 and neg <= 20 and (avg is None or avg >= 4.0):
        label, tone = "Worth it to buy", "good"
        reason = f"{pos}% reviews are positive and only {neg}% are negative."
    elif neg >= 35 or (avg is not None and avg < 3.2):
        label, tone = "Not recommended", "bad"
        reason = f"{neg}% reviews are negative"
        if avg is not None:
            reason += f" and the average rating is {avg}/5"
        reason += "."
    else:
        label, tone = "Average, think before buying", "mid"
        reason = f"Mixed opinions: {pos}% positive, {neg}% negative."

    if total < 10:
        reason += " Very few reviews, so this result is not reliable."
    return {"label": label, "tone": tone, "reason": reason}