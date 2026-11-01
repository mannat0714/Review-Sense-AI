import pandas as pd

df = pd.read_csv("1429_1.csv", low_memory=False)

# sirf 2 columns lo aur khali rows hata do
df = df[["reviews.text", "reviews.rating"]].dropna()

# rating se sentiment banao
def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["reviews.rating"].apply(get_sentiment)

# har sentiment me se 800 reviews random lo
balanced = df.groupby("sentiment").sample(n=800, random_state=42)

# columns ke naam simple karo
balanced = balanced[["reviews.text", "sentiment"]]
balanced.columns = ["review", "sentiment"]

# file me save karo
balanced.to_csv("balanced_reviews.csv", index=False)

print(balanced["sentiment"].value_counts())
print("File save ho gayi: balanced_reviews.csv")