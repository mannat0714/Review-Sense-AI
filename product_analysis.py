"""/product page: paste a product link, pick from dropdown, or paste reviews -> sentiment + buy verdict."""
import os
import re

import pandas as pd
from flask import render_template, request

from reviews_fetcher import get_reviews
from verdict import analyse, buy_verdict

DATASET_FILE = "1429_1.csv"
MAX_PASTED = 500          # max number of pasted reviews analysed

SOURCE_LABELS = {
    "dataset": "Dataset (Kaggle Amazon reviews)",
    "api": "Live reviews via Review API",
    "scraping": "Live scraping (only top reviews on the page)",
    "pasted": "Reviews pasted by the user",
}


def _pick(columns, candidates):
    lowered = {str(c).lower().strip(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def parse_pasted(raw):
    """Turn the pasted text into a list of review dicts.

    If the text has blank lines, each blank-line separated block is one review
    (so a review can span several lines). Otherwise every line is one review.
    """
    raw = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []
    if re.search(r"\n\s*\n", raw):
        chunks = re.split(r"\n\s*\n", raw)
    else:
        chunks = raw.split("\n")
    items = []
    for chunk in chunks:
        text = " ".join(chunk.split())
        if len(text) >= 3:
            items.append({"text": text, "rating": None})
    return items[:MAX_PASTED]


def _load_dataset(path):
    """Return (by_asin, by_name) dicts of review lists. Empty dicts if file missing."""
    by_asin, by_name = {}, {}
    if not os.path.exists(path):
        return by_asin, by_name
    df = pd.read_csv(path, encoding_errors="replace", low_memory=False)

    c_text = _pick(df.columns, ["reviews.text", "review", "reviews_text", "text"])
    c_title = _pick(df.columns, ["reviews.title", "title"])
    c_rate = _pick(df.columns, ["reviews.rating", "rating"])
    c_asin = _pick(df.columns, ["asins", "asin"])
    c_name = _pick(df.columns, ["name", "product_name"])
    if c_text is None:
        return by_asin, by_name

    df = df.dropna(subset=[c_text])
    for _, row in df.iterrows():
        text = str(row[c_text]).strip()
        if c_title is not None and pd.notna(row[c_title]):
            text = f"{str(row[c_title]).strip()}. {text}"
        if not text:
            continue
        rating = row[c_rate] if c_rate is not None and pd.notna(row[c_rate]) else None
        item = {"text": text, "rating": rating}

        if c_asin is not None and pd.notna(row[c_asin]):
            for a in str(row[c_asin]).split(","):
                by_asin.setdefault(a.strip().upper(), []).append(item)
        if c_name is not None and pd.notna(row[c_name]):
            by_name.setdefault(str(row[c_name]).strip(), []).append(item)
    return by_asin, by_name


def register_product_routes(app, model):
    by_asin, by_name = _load_dataset(os.path.join(app.root_path, DATASET_FILE))

    # dropdown: products with the most reviews first
    product_options = sorted(by_name.keys(), key=lambda n: len(by_name[n]), reverse=True)

    def lookup_by_asin(asin):
        return by_asin.get(asin.upper(), [])

    @app.route("/product", methods=["GET", "POST"])
    def product():
        result = error = None
        link = selected = pasted = ""

        if request.method == "POST":
            link = request.form.get("link", "").strip()
            selected = request.form.get("product", "").strip()
            pasted = request.form.get("pasted", "")
            pasted_reviews = parse_pasted(pasted)
            reviews, source, asin, name = [], None, None, None

            if link:
                reviews, source, asin = get_reviews(link, dataset_lookup=lookup_by_asin)
                if source == "invalid_link":
                    if pasted_reviews:
                        reviews, source = pasted_reviews, "pasted"
                    else:
                        error = ("This does not look like an Amazon product link. Paste a link containing "
                                 "/dp/ASIN, pick a product from the list, or paste the reviews yourself.")
                elif not reviews:
                    if pasted_reviews:
                        # link could not be read (login / captcha), use the pasted reviews instead
                        reviews, source = pasted_reviews, "pasted"
                    else:
                        error = ("Could not get reviews for this link. Amazon may require login or show a "
                                 "captcha, and the product is not in the dataset. Copy the reviews from the "
                                 "product page and paste them in the box below, or pick a product from the list.")
            elif pasted_reviews:
                reviews, source = pasted_reviews, "pasted"
            elif selected:
                reviews, source, name = by_name.get(selected, []), "dataset", selected
                if not reviews:
                    error = "No reviews found for this product."
            else:
                error = "Paste a product link, choose a product, or paste some reviews first."

            if reviews and not error:
                res = analyse(model, reviews)
                if res is None:
                    error = "No readable review text found for this product."
                else:
                    res["verdict"] = buy_verdict(res["pct"], res["avg_rating"], res["total"])
                    res["source"] = SOURCE_LABELS.get(source, source)
                    res["asin"] = asin
                    res["name"] = name
                    result = res

        return render_template(
            "product.html", result=result, error=error,
            link=link, selected=selected, pasted=pasted, products=product_options,
        )