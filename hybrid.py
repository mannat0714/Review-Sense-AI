"""Hybrid sentiment: ML model + common-sense word rules (lexicon).

Why: a TF-IDF model has very little to work with in tiny reviews such as
"this is good" or "average". For short reviews we therefore also check a
small sentiment word list (like VADER, but self-made, no extra install).

Rules (see combine()):
  1. Short review (8 words or fewer) and the word list disagrees with the
     model  ->  trust the word list.
  2. Longer review, model is unsure (confidence < 55%) and the word list is
     very strong  ->  trust the word list.
  3. Otherwise the ML model decides.
"""
import math
import re

import numpy as np

from model_insights import confidence_for

SHORT_REVIEW_WORDS = 8
POLAR_THRESHOLD = 0.35        # |compound| needed for Positive / Negative

# ---------------------------------------------------------------- word lists
POS = {
    "good": 1.9, "great": 2.5, "excellent": 3.0, "amazing": 2.8, "awesome": 2.8,
    "fantastic": 2.9, "wonderful": 2.7, "perfect": 2.8, "love": 2.6, "loved": 2.6,
    "nice": 1.8, "best": 2.7, "happy": 2.2, "satisfied": 2.0, "superb": 3.0,
    "brilliant": 2.8, "outstanding": 3.0, "impressive": 2.2, "recommend": 1.8,
    "recommended": 1.8, "worth": 1.5, "smooth": 1.5, "comfortable": 1.8,
    "durable": 1.8, "reliable": 2.0, "sturdy": 1.6, "beautiful": 2.4, "pleased": 2.0,
    "enjoy": 2.0, "enjoyed": 2.0, "favorite": 2.2, "favourite": 2.2, "helpful": 1.8,
    "easy": 1.3, "useful": 1.8, "cool": 1.5, "better": 1.2, "affordable": 1.3,
    "fast": 1.0, "stylish": 1.8, "lovely": 2.4, "glad": 1.8, "like": 1.0,
}
NEG = {
    "bad": 2.5, "terrible": 3.0, "awful": 3.0, "horrible": 3.0, "worst": 3.2,
    "poor": 2.3, "useless": 2.8, "waste": 2.4, "disappointed": 2.5,
    "disappointing": 2.5, "hate": 2.8, "hated": 2.8, "broken": 2.3, "broke": 2.2,
    "defective": 2.8, "faulty": 2.5, "slow": 1.5, "fake": 2.6, "fraud": 3.0,
    "scam": 3.0, "regret": 2.5, "annoying": 2.0, "overpriced": 2.0, "expensive": 1.2,
    "damaged": 2.3, "problem": 1.5, "problems": 1.6, "issue": 1.2, "issues": 1.3,
    "failed": 2.2, "worthless": 2.8, "pathetic": 2.8, "rubbish": 2.7, "garbage": 2.8,
    "trash": 2.7, "unhappy": 2.3, "dissatisfied": 2.4, "cheap": 0.8, "ugly": 2.2,
    "dead": 1.8, "stopped": 1.0, "sucks": 2.6, "lame": 1.8,
    "late": 1.8, "delayed": 2.0, "delay": 1.6, "pricey": 1.8, "costly": 1.8,
    "lack": 2.0, "lacking": 2.0, "missing": 1.8, "wrong": 2.0, "leaking": 2.2,
    "leak": 2.0, "overheating": 2.5, "heating": 1.8, "noisy": 1.8, "weak": 1.8,
    "flimsy": 2.2, "unreliable": 2.4, "inferior": 2.4,
}
# "high" / "low" depend on context: high price = bad, high quality = good,
# low price = good, low quality = bad
PRICE_WORDS = {"price", "prices", "cost", "costs", "priced", "charges", "rate", "fee", "fees"}
NEUTRAL = {
    "average", "okay", "ok", "okayish", "fine", "decent", "mediocre", "normal",
    "acceptable", "alright", "ordinary", "moderate", "passable", "so-so", "meh",
}
INTENSIFIERS = {
    "very": 0.3, "really": 0.3, "extremely": 0.5, "so": 0.25, "too": 0.25,
    "absolutely": 0.4, "super": 0.4, "highly": 0.4, "totally": 0.3, "quite": 0.15,
    "slightly": -0.3, "somewhat": -0.25, "bit": -0.25,
}
NEGATIONS = {
    "not", "no", "never", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "isn't", "isnt", "wasn't", "wasnt", "cannot", "can't", "cant", "won't", "wont",
    "hardly", "neither",
}
# multi-word phrases: value > 0 positive, < 0 negative, None = neutral
PHRASES = {
    "waste of money": -2.8, "not worth": -2.2, "worth it": 2.2, "worth the money": 2.4,
    "value for money": 2.2, "highly recommend": 3.0, "do not buy": -3.0,
    "don't buy": -3.0, "dont buy": -3.0, "not recommend": -2.6, "not recommended": -2.6,
    "stopped working": -2.8, "does not work": -2.6, "doesn't work": -2.6,
    "doesnt work": -2.6, "not working": -2.6, "must buy": 2.5, "go for it": 2.0,
    "nothing special": None, "could be better": None, "not great": -2.2,
    "not bad": None, "just ok": None, "just okay": None,
}
_PHRASE_RES = [
    (re.compile(r"\b" + re.escape(p) + r"\b"), p, v)
    for p, v in sorted(PHRASES.items(), key=lambda kv: -len(kv[0]))
]


# ------------------------------------------------------------------- scoring
def score(text):
    """Return (compound, label_or_None, evidence_words, word_count)."""
    t = str(text).lower().replace("\u2019", "'")
    word_count = len(re.findall(r"[a-z']+", t))
    total, hits, neutral_hits = 0.0, [], []

    for regex, phrase, val in _PHRASE_RES:
        def _take(m, phrase=phrase, val=val):
            nonlocal total
            hits.append(phrase)
            if val is None:
                neutral_hits.append(phrase)
            else:
                total += val
            return " "
        t = regex.sub(_take, t)

    tokens = re.findall(r"[a-z'\-]+", t)
    for i, tok in enumerate(tokens):
        if tok in NEUTRAL:
            neutral_hits.append(tok)
            hits.append(tok)
            continue
        if tok in ("high", "low"):
            near = tokens[max(0, i - 3):i + 4]
            about_price = any(w in PRICE_WORDS for w in near)
            if tok == "high":
                val = -2.0 if about_price else 1.6
            else:
                val = 1.8 if about_price else -2.0
        elif tok in POS:
            val = POS[tok]
        elif tok in NEG:
            val = -NEG[tok]
        else:
            continue
        window = tokens[max(0, i - 3):i]
        boost = sum(INTENSIFIERS.get(w, 0) for w in window)
        negated = any(w in NEGATIONS for w in window)
        val *= (1 + boost)
        if negated:
            val = -val * 0.9 if val > 0 else -val * 0.5
        total += val
        hits.append(tok)

    compound = total / math.sqrt(total * total + 15)
    if compound >= POLAR_THRESHOLD:
        label = "Positive"
    elif compound <= -POLAR_THRESHOLD:
        label = "Negative"
    elif hits:
        label = "Neutral"
    else:
        label = None
    evidence = []
    for h in hits:
        if h not in evidence:
            evidence.append(h)
    return compound, label, evidence[:4], word_count


def combine(text, model_label, model_conf):
    """Return (label, confidence, method, evidence). method: 'model' or 'rules'."""
    compound, lex_label, evidence, n_words = score(text)

    if lex_label is not None and lex_label != model_label:
        short = n_words <= SHORT_REVIEW_WORDS
        unsure_model = (model_conf is not None and model_conf < 55
                        and abs(compound) >= 0.6)
        if short or unsure_model:
            conf = 65.0 if lex_label == "Neutral" else round(min(95.0, 60 + abs(compound) * 45), 1)
            return lex_label, conf, "rules", evidence
    return model_label, model_conf, "model", []


# ------------------------------------------------------------ model wrappers
def decide_single(model, text):
    """One text -> (label, confidence, method, evidence)."""
    label = str(model.predict([text])[0])
    return combine(text, label, confidence_for(model, text))


def _batch_conf(model, texts):
    try:
        if hasattr(model, "predict_proba"):
            return list(np.asarray(model.predict_proba(texts)).max(axis=1) * 100)
        if hasattr(model, "decision_function"):
            s = np.asarray(model.decision_function(texts), dtype=float)
            if s.ndim == 2 and s.shape[1] > 1:
                z = s * 3.0
                e = np.exp(z - z.max(axis=1, keepdims=True))
                return list(e.max(axis=1) / e.sum(axis=1) * 100)
    except Exception:
        pass
    return [None] * len(texts)


class HybridModel:
    """Drop-in wrapper: has .predict() like the sklearn model, but applies the
    word rules on top. Used for batch and product analysis."""

    def __init__(self, model):
        self.model = model

    def predict(self, texts):
        texts = [str(t) for t in texts]
        preds = [str(p) for p in self.model.predict(texts)]
        confs = _batch_conf(self.model, texts)
        return [combine(t, p, c)[0] for t, p, c in zip(texts, preds, confs)]

    def __getattr__(self, name):          # classes_, steps, ... come from the real model
        return getattr(self.model, name)