# task-2/sentiment_analysis.py

import pandas as pd
from textblob import TextBlob
from typing import Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer

class SentimentAnalyzer:
    """
    Performs sentiment analysis and keyword-based thematic assignment
    on preprocessed bank app reviews.
    """

    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        self.df = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.df)} preprocessed reviews from {self.input_file}")

    @staticmethod
    def compute_sentiment(text: str) -> str:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'

    @staticmethod
    def compute_sentiment_score(text: str) -> float:
        return TextBlob(text).sentiment.polarity

    def assign_themes(self, text: str, bank: str) -> List[str]:
        """
        Assign all matching themes to a review based on keywords for the bank.
        Returns a list of themes, or ["Other"] if none matched.
        """
        theme_keywords = {
            "CBE": {
                "Login Issues": ["login", "signin", "authentication"],
                "Transaction Speed": ["slow transfer", "lag", "slow"],
                "UI Experience": ["ui", "interface", "design"]
            },
            "BOA": {
                "App Crashes": ["crash", "error", "bug"],
                "Transfer Delays": ["slow transfer", "delay"],
                "Customer Support": ["support", "help", "service"]
            },
            "Dashen": {
                "Login Errors": ["login", "signin", "authentication"],
                "Slow Performance": ["slow", "lag", "hang"],
                "Feature Requests": ["feature", "add", "request"]
            }
        }

        matched_themes = []
        for theme, keywords in theme_keywords.get(bank, {}).items():
            for kw in keywords:
                if kw.lower() in text.lower():
                    matched_themes.append(theme)
                    break  # avoid duplicate themes per keyword list
        return matched_themes if matched_themes else ["Other"]

    def analyze_sentiments(self) -> None:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Compute sentiment
        self.df['sentiment'] = self.df['review_text'].apply(self.compute_sentiment)
        self.df['sentiment_score'] = self.df['review_text'].apply(self.compute_sentiment_score)
        print("Sentiment analysis completed.")

        # Assign themes based on keywords
        self.df['themes'] = self.df.apply(lambda row: self.assign_themes(row['review_text'], row['bank']), axis=1)
        print("Theme assignment completed.")

        # Optional: TF-IDF top keywords per bank (for reporting)
        banks = self.df['bank'].unique()
        top_n = 10
        for bank in banks:
            bank_reviews = self.df[self.df['bank'] == bank]['review_text'].astype(str).tolist()
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
            X = vectorizer.fit_transform(bank_reviews)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = X.sum(axis=0).A1
            word_score = dict(zip(feature_names, tfidf_scores))
            top_keywords = sorted(word_score.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_words = [k for k, v in top_keywords]
            print(f"\n{bank} Top Keywords: {top_words}")

    def save_results(self) -> None:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Convert themes list to comma-separated string for CSV
        self.df['themes'] = self.df['themes'].apply(lambda x: ", ".join(x))
        self.df.to_csv(self.output_file, index=False)
        print(f"Sentiment & theme results saved to {self.output_file}")


if __name__ == "__main__":
    analyzer = SentimentAnalyzer(
        input_file='task-1/clean_reviews.csv',
        output_file='task-2/reviews_sentiment_themes.csv'
    )
    analyzer.load_data()
    analyzer.analyze_sentiments()
    analyzer.save_results()
