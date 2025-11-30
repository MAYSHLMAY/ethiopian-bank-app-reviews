# task-2/sentiment_analysis.py

import pandas as pd
from textblob import TextBlob
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer


class SentimentAnalyzer:
    """
    A class to perform sentiment analysis on preprocessed bank app reviews.
    Uses TextBlob to classify reviews into positive, negative, or neutral sentiment,
    and extracts keywords/themes for interim reporting.
    """

    def __init__(self, input_file: str, output_file: str):
        """
        Initializes the sentiment analyzer with input and output file paths.

        Args:
            input_file (str): Path to the preprocessed reviews CSV.
            output_file (str): Path to save the reviews with sentiment labels and themes.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """
        Load preprocessed reviews from a CSV file into a DataFrame.
        """
        self.df = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.df)} preprocessed reviews from {self.input_file}")

    @staticmethod
    def compute_sentiment(text: str) -> str:
        """
        Compute sentiment for a single review using TextBlob.

        Args:
            text (str): Review text.

        Returns:
            str: Sentiment label - 'positive', 'negative', or 'neutral'.
        """
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'

    def analyze_sentiments(self) -> None:
        """
        Apply sentiment analysis to all reviews in the DataFrame and extract themes.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Sentiment
        self.df['sentiment'] = self.df['review_text'].apply(self.compute_sentiment)
        print("Sentiment analysis completed.")

        # Keyword extraction per bank using TF-IDF
        banks = self.df['bank'].unique()
        top_n = 10
        self.df['theme'] = None  # Initialize theme column

        # Define manual theme mapping (interim)
        theme_mapping = {
            "CBE": ["Login Issues", "Transaction Speed", "UI Experience"],
            "BOA": ["App Crashes", "Transfer Delays", "Customer Support"],
            "Dashen": ["Login Errors", "Slow Performance", "Feature Requests"]
        }

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
            print(f"{bank} Themes: {theme_mapping[bank]}")

            # Assign all reviews for this bank the first theme for simplicity
            self.df.loc[self.df['bank'] == bank, 'theme'] = theme_mapping[bank][0]

    def save_results(self) -> None:
        """
        Save the DataFrame with sentiment labels and themes to a CSV file.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.df.to_csv(self.output_file, index=False)
        print(f"Sentiment & theme results saved to {self.output_file}")


if __name__ == "__main__":
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(
        input_file='task-1/clean_reviews.csv',
        output_file='task-2/reviews_sentiment_themes.csv'
    )

    # Run sentiment + theme pipeline
    analyzer.load_data()
    analyzer.analyze_sentiments()
    analyzer.save_results()
