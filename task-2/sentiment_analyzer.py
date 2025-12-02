# task-2/sentiment_analyzer.py
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class SentimentAnalyzer:
    def __init__(self, input_csv="task-1/clean_reviews.csv", output_csv="task-2/reviews_sentiment_themes.csv"):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = pd.read_csv(input_csv)
        self.themes_per_bank = {}
        # Words to ignore for better themes
        self.stop_words = ["app", "good", "bank", "mobile", "use", "using"]

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def analyze_sentiment(self):
        """
        Compute sentiment polarity for each review using TextBlob.
        """
        self.df["sentiment_score"] = self.df["review_text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        print("Sentiment analysis completed.")

    def extract_themes(self, top_n_keywords=5):
        """
        Extract top keywords per bank using TF-IDF and map to broad themes.
        """
        banks = self.df["bank"].unique()
        for bank in banks:
            bank_df = self.df[self.df["bank"] == bank]
            # Preprocess review text
            corpus = bank_df["review_text"].apply(self.clean_text).tolist()
            tfidf = TfidfVectorizer(max_features=50, stop_words="english")
            X = tfidf.fit_transform(corpus)
            feature_names = tfidf.get_feature_names_out()
            # Remove stop words
            feature_names = [w for w in feature_names if w not in self.stop_words]
            # Take top_n_keywords
            top_keywords = feature_names[:top_n_keywords]
            self.themes_per_bank[bank] = top_keywords
        print(f"Thematic extraction completed. Themes per bank: {self.themes_per_bank}")

        # Assign themes to each review
        def assign_theme(review_text, bank):
            text = self.clean_text(review_text)
            keywords = self.themes_per_bank.get(bank, [])
            matched = [kw for kw in keywords if kw in text]
            return matched if matched else ["Other"]

        self.df["themes"] = self.df.apply(lambda row: assign_theme(row["review_text"], row["bank"]), axis=1)

    def save_csv(self):
        """
        Save the updated DataFrame with sentiment and themes.
        """
        self.df.to_csv(self.output_csv, index=False)
        print(f"Updated CSV saved → {self.output_csv}")

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print(f"Loaded {len(analyzer.df)} reviews from {analyzer.input_csv}")
    analyzer.analyze_sentiment()
    analyzer.extract_themes(top_n_keywords=5)
    analyzer.save_csv()
