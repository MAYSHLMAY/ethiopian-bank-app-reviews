# task-4/analyze_reviews.py

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional

class ReviewAnalyzer:
    """
    Connects to PostgreSQL and performs analysis & reporting on bank reviews.
    Includes summary stats, sentiment/theme distributions, time-series, top keywords,
    and heatmaps for deeper insights.
    """
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.conn: Optional[psycopg2.extensions.connection] = None
        self.df: Optional[pd.DataFrame] = None

    def connect_db(self) -> None:
        self.conn = psycopg2.connect(**self.db_config)
        print(f"Connected to PostgreSQL database: {self.db_config['dbname']}")

    def load_data(self) -> None:
        query = """
            SELECT r.review_id, b.bank_name, b.app_name, r.review_text, r.rating,
                   r.review_date, r.sentiment_label, r.sentiment_score, r.theme, r.source
            FROM reviews r
            JOIN banks b ON r.bank_id = b.bank_id
        """
        self.df = pd.read_sql(query, self.conn)
        print(f"Loaded {len(self.df)} reviews from database")

    def aggregate_metrics(self) -> pd.DataFrame:
        review_count = self.df.groupby('bank_name')['review_id'].count().reset_index()
        review_count.rename(columns={'review_id': 'review_count'}, inplace=True)
        avg_rating = self.df.groupby('bank_name')['rating'].mean().reset_index()
        avg_rating.rename(columns={'rating': 'avg_rating'}, inplace=True)
        summary = pd.merge(review_count, avg_rating, on='bank_name')
        return summary

    # ------------------------- PLOTTING FUNCTIONS -------------------------

    def sentiment_distribution(self) -> None:
        """Bar chart showing sentiment distribution per bank"""
        plt.figure(figsize=(12,6))
        sns.set_style("whitegrid")
        palette = sns.color_palette("Paired")  # distinct sentiment colors

        sns.countplot(
            data=self.df,
            x='bank_name',
            hue='sentiment_label',
            palette=palette,
            dodge=True
        )

        plt.title("Sentiment Distribution per Bank", fontsize=18, fontweight='bold')
        plt.xlabel("Bank", fontsize=14)
        plt.ylabel("Number of Reviews", fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Sentiment', title_fontsize=13, fontsize=11, loc='upper right')
        plt.tight_layout()
        plt.savefig("task-4/sentiment_distribution.png", dpi=300)
        plt.close()
        print("Sentiment distribution plot saved.")

    def theme_distribution(self) -> None:
        """Bar chart showing theme distribution per bank"""
        self.df['theme'] = self.df['theme'].fillna('Other')
        exploded = self.df.assign(theme=self.df['theme'].str.split(', ')).explode('theme')

        plt.figure(figsize=(14,8))
        sns.set_style("whitegrid")
        palette = sns.color_palette("Set2")  # soft, visually pleasant colors

        sns.countplot(
            data=exploded,
            x='bank_name',
            hue='theme',
            palette=palette,
            dodge=True
        )

        plt.title("Theme Distribution per Bank", fontsize=18, fontweight='bold')
        plt.xlabel("Bank", fontsize=14)
        plt.ylabel("Number of Reviews", fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Theme', title_fontsize=13, fontsize=11, loc='upper right')
        plt.tight_layout()
        plt.savefig("task-4/theme_distribution.png", dpi=300)
        plt.close()
        print("Theme distribution plot saved.")

    def time_series_analysis(self) -> None:
        """Line chart showing review volume over time for each bank"""
        self.df['review_date'] = pd.to_datetime(self.df['review_date'])
        plt.figure(figsize=(14,6))
        sns.set_style("whitegrid")
        colors = sns.color_palette("tab10", n_colors=len(self.df['bank_name'].unique()))

        for idx, bank in enumerate(self.df['bank_name'].unique()):
            bank_df = self.df[self.df['bank_name'] == bank].groupby('review_date').size()
            bank_df.plot(label=bank, color=colors[idx], linewidth=2, marker='o')

        plt.title("Review Volume Over Time", fontsize=18, fontweight='bold')
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Number of Reviews", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Bank', title_fontsize=13, fontsize=11)
        plt.tight_layout()
        plt.savefig("task-4/review_timeseries.png", dpi=300)
        plt.close()
        print("Time-series plot saved.")

    def top_keywords(self, top_n: int = 10) -> None:
        """TF-IDF top keywords per bank"""
        for bank in self.df['bank_name'].unique():
            bank_reviews = self.df[self.df['bank_name'] == bank]['review_text'].astype(str).tolist()
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
            X = vectorizer.fit_transform(bank_reviews)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = X.sum(axis=0).A1
            word_score = dict(zip(feature_names, tfidf_scores))
            top_keywords = sorted(word_score.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_words = [k for k, v in top_keywords]
            print(f"\n{bank} Top {top_n} Keywords: {top_words}")

    def rating_sentiment_heatmap(self) -> None:
        """Heatmap showing rating vs sentiment"""
        pivot = self.df.pivot_table(index='rating', columns='sentiment_label', aggfunc='size', fill_value=0)
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
        plt.title("Heatmap: Star Rating vs Sentiment", fontsize=16, fontweight='bold')
        plt.ylabel("Rating", fontsize=12)
        plt.xlabel("Sentiment", fontsize=12)
        plt.tight_layout()
        plt.savefig("task-4/rating_sentiment_heatmap.png", dpi=300)
        plt.close()
        print("Rating vs Sentiment heatmap saved.")

    def theme_sentiment_heatmap(self) -> None:
        """Heatmap showing theme vs sentiment"""
        exploded = self.df.assign(theme=self.df['theme'].str.split(', ')).explode('theme')
        pivot = exploded.pivot_table(index='theme', columns='sentiment_label', aggfunc='size', fill_value=0)
        plt.figure(figsize=(12,8))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="coolwarm", linewidths=.5)
        plt.title("Heatmap: Theme vs Sentiment", fontsize=16, fontweight='bold')
        plt.ylabel("Theme", fontsize=12)
        plt.xlabel("Sentiment", fontsize=12)
        plt.tight_layout()
        plt.savefig("task-4/theme_sentiment_heatmap.png", dpi=300)
        plt.close()
        print("Theme vs Sentiment heatmap saved.")

    def save_summary(self, summary: pd.DataFrame) -> None:
        summary.to_csv("task-4/bank_review_summary.csv", index=False)
        print("Summary CSV saved.")

    # ------------------------- UTILITY -------------------------

    def close_db(self) -> None:
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    # ------------------------- RUN PIPELINE -------------------------

    def run(self) -> None:
        self.connect_db()
        self.load_data()
        summary = self.aggregate_metrics()
        self.save_summary(summary)
        self.sentiment_distribution()
        self.theme_distribution()
        self.time_series_analysis()
        self.top_keywords()
        self.rating_sentiment_heatmap()
        self.theme_sentiment_heatmap()
        self.close_db()


if __name__ == "__main__":
    db_config = {
        "host": "localhost",
        "port": 5432,
        "dbname": "bank_reviews",
        "user": "postgres",
        "password": "1234"
    }

    analyzer = ReviewAnalyzer(db_config)
    analyzer.run()
