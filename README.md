# Review-Sense-AI

*Product Review Sentiment Analysis*: A web application that reads customer reviews and tells you whether a product is *worth buying* or not.

## About the Project

Online shopping sites have thousands of reviews, and nobody can read them all. Review-Sense-AI uses Machine Learning to analyze reviews, classify them as **Positive** or **Negative**, and give a final **verdict** on whether the product is worth buying.

## Features

- Sentiment analysis of product reviews (Positive / Negative)
- Paste a product link and fetch its reviews
- Compare reviews across different shopping platforms
- Final verdict: "Worth it" or "Not worth it"
- Simple, easy-to-use website built with Flask

## Tech Stack

- **Language:** Python
- **Web framework:** Flask
- **ML / NLP:** scikit-learn, pandas, TF-IDF vectorizer
- **Model:**  Logistic Regression / Naive Bayes / SVM
- **Frontend:** HTML, CSS (`templates` and `static` folders)

## Dataset

- Flipkart product reviews (Kaggle)
- Amazon product reviews (`1429_1.csv`)
- The dataset was balanced (`balanced_reviews.csv`) before training the model

## Project Structure

```
Review-Sense-AI/
├── app.py                # Main Flask app
├── train_model.py        # Model training
├── compare_models.py     # Comparing different models
├── reviews_fetcher.py    # Fetching reviews
├── verdict.py            # Final verdict logic
├── models/               # Saved trained models
├── templates/            # HTML pages
├── static/               # CSS, images
└── requirements.txt      # Required libraries
```

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/mannat0714/Review-Sense-AI.git
   cd Review-Sense-AI
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   python app.py
   ```
5. Open in your browser: `http://127.0.0.1:5000`

## Model Accuracy

The Flipkart model reaches about **84%** accuracy. [Update this if you have a newer model or accuracy.]

## Author

**Mannat Kushwaha**
B.Tech CSE (AI & ML), Lloyd Institute of Engineering & Technology
Guide: Ms. Priyanka Rahi, Assistant Professor, CSE
