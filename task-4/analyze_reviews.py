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
    heatmaps, narrative insights, and recommendations.
    """
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.conn: Optional[psycopg2.extensions.connection] = None
        self.df: Optional[pd.DataFrame] = None

    # ------------------------- DATABASE -------------------------

    def connect_db(self) -> None:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print(f"Connected to PostgreSQL database: {self.db_config['dbname']}")
        except Exception as e:
            print(f"Error connecting to DB: {e}")
            raise

    def load_data(self) -> None:
        try:
            query = """
                SELECT r.review_id, b.bank_name, b.app_name, r.review_text, r.rating,
                       r.review_date, r.sentiment_label, r.sentiment_score, r.theme, r.source
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
            """
            self.df = pd.read_sql(query, self.conn)
            print(f"Loaded {len(self.df)} reviews from database")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    # ------------------------- METRICS -------------------------

    def aggregate_metrics(self) -> pd.DataFrame:
        review_count = self.df.groupby('bank_name')['review_id'].count().reset_index()
        review_count.rename(columns={'review_id': 'review_count'}, inplace=True)
        avg_rating = self.df.groupby('bank_name')['rating'].mean().reset_index()
        avg_rating.rename(columns={'rating': 'avg_rating'}, inplace=True)
        summary = pd.merge(review_count, avg_rating, on='bank_name')
        return summary

    # ------------------------- PLOTTING FUNCTIONS -------------------------

    def sentiment_distribution(self) -> None:
        try:
            plt.figure(figsize=(12,6))
            sns.set_style("whitegrid")
            palette = sns.color_palette("Paired")

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
        except Exception as e:
            print(f"Error generating sentiment distribution plot: {e}")

    def theme_distribution(self) -> None:
        try:
            self.df['theme'] = self.df['theme'].fillna('Other')
            exploded = self.df.assign(theme=self.df['theme'].str.split(', ')).explode('theme')

            plt.figure(figsize=(14,8))
            sns.set_style("whitegrid")
            palette = sns.color_palette("Set2")

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
        except Exception as e:
            print(f"Error generating theme distribution plot: {e}")

    def time_series_analysis(self) -> None:
        try:
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
        except Exception as e:
            print(f"Error generating time-series plot: {e}")

    def top_keywords(self, top_n: int = 10) -> None:
        try:
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
        except Exception as e:
            print(f"Error generating top keywords: {e}")

    def rating_sentiment_heatmap(self) -> None:
        try:
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
        except Exception as e:
            print(f"Error generating rating vs sentiment heatmap: {e}")

    def theme_sentiment_heatmap(self) -> None:
        try:
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
        except Exception as e:
            print(f"Error generating theme vs sentiment heatmap: {e}")

    # ------------------------- INSIGHTS & RECOMMENDATIONS -------------------------

    def narrative_insights(self) -> None:
        """
        Generate explicit insights and actionable recommendations
        for each bank based on sentiment and theme analysis.
        """
        if self.df is None or self.df.empty:
            print("No data loaded for insights.")
            return

        for bank in self.df['bank_name'].unique():
            bank_df = self.df[self.df['bank_name'] == bank]
            if bank_df.empty:
                continue

            pos_themes = bank_df[bank_df['sentiment_label']=='positive']['theme'].value_counts().head(3)
            neg_themes = bank_df[bank_df['sentiment_label']=='negative']['theme'].value_counts().head(3)

            print(f"\n--- {bank} ---")
            print("Top Positive Themes / Drivers:")
            for t, count in pos_themes.items():
                print(f"- {t}: {count} mentions")

            print("Top Negative Themes / Pain Points:")
            for t, count in neg_themes.items():
                print(f"- {t}: {count} mentions")

            recs = []
            if 'slow transfer' in neg_themes.index:
                recs.append("Optimize transfer processing speed.")
            if 'login issues' in neg_themes.index:
                recs.append("Enhance login stability (2FA/biometrics).")
            if 'UI' in neg_themes.index:
                recs.append("Redesign confusing UI elements.")
            if 'crash' in neg_themes.index:
                recs.append("Investigate app crashes and improve stability.")

            if recs:
                print("Recommendations:")
                for r in recs:
                    print(f"- {r}")
            else:
                print("No major issues detected. Continue monitoring user feedback.")

    # ------------------------- UTILITY -------------------------

    def save_summary(self, summary: pd.DataFrame) -> None:
        summary.to_csv("task-4/bank_review_summary.csv", index=False)
        print("Summary CSV saved.")

    def close_db(self) -> None:
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    # ------------------------- RUN PIPELINE -------------------------

    def run(self) -> None:
        try:
            self.connect_db()
            self.load_data()
        except Exception as e:
            print(f"Error connecting/loading data: {e}")
            return

        summary = self.aggregate_metrics()
        self.save_summary(summary)

        # Plotting functions with safeguards
        for plot_func in [
            self.sentiment_distribution,
            self.theme_distribution,
            self.time_series_analysis,
            self.rating_sentiment_heatmap,
            self.theme_sentiment_heatmap
        ]:
            try:
                plot_func()
            except Exception as e:
                print(f"Error generating plot {plot_func.__name__}: {e}")

        try:
            self.top_keywords()
        except Exception as e:
            print(f"Error generating top keywords: {e}")

        # Narrative insights and recommendations
        self.narrative_insights()

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
