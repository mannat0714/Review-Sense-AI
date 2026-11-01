import pandas as pd

# 1) Load the Amazon reviews file (put this script in the same folder as 1429_1.csv)
df = pd.read_csv("1429_1.csv")
df = df[["reviews.text", "reviews.rating"]].dropna()
df.columns = ["review", "rating"]

# 2) Clean basic problems
df["rating"] = pd.to_numeric(df["rating"], errors="coerce").round()
df = df.dropna(subset=["rating"])
df = df[df["rating"].between(1, 5)]
df["review"] = df["review"].astype(str).str.strip()
df = df[df["review"].str.len() > 3].drop_duplicates(subset="review")


# 3) Rating -> sentiment label
def to_label(r):
    if r <= 2:
        return "Negative"
    if r == 3:
        return "Neutral"
    return "Positive"


df["sentiment"] = df["rating"].apply(to_label)
print("Before balancing:")
print(df["sentiment"].value_counts(), "\n")

# 4) Balance: take the same number of reviews from every class
n = df["sentiment"].value_counts().min()
parts = [df[df["sentiment"] == s].sample(n, random_state=42)
         for s in ["Negative", "Neutral", "Positive"]]
balanced = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

print("After balancing:")
print(balanced["sentiment"].value_counts())

# 5) Save
balanced[["review", "sentiment"]].to_csv("balanced_reviews.csv", index=False)
print("\nSaved balanced_reviews.csv with", len(balanced), "rows")