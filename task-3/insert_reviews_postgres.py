# task-3/insert_reviews_postgres.py

import psycopg2
import pandas as pd
from textblob import TextBlob
from typing import Optional, Dict


class PostgresReviewInserter:
    """
    Inserts bank app reviews from a CSV file into a PostgreSQL database.

    Attributes:
        db_config (dict): Dictionary with PostgreSQL connection parameters.
        csv_path (str): Path to the CSV file containing reviews.
        bank_mapping (dict): Mapping from CSV bank codes to database bank names.
    """

    def __init__(self, db_config: Dict[str, str], csv_path: str, bank_mapping: Optional[Dict[str, str]] = None):
        self.db_config = db_config
        self.csv_path = csv_path
        self.bank_mapping = bank_mapping or {
            "CBE": "Commercial Bank of Ethiopia",
            "BOA": "Bank of Abyssinia",
            "Dashen": "Dashen Bank"
        }
        self.conn: Optional[psycopg2.extensions.connection] = None
        self.cursor: Optional[psycopg2.extensions.cursor] = None
        self.df: Optional[pd.DataFrame] = None

    def connect_db(self) -> None:
        """Connect to PostgreSQL database."""
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor()
        print(f"Connected to PostgreSQL database: {self.db_config.get('dbname')}")

    def load_csv(self) -> None:
        """Load CSV reviews into pandas DataFrame."""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} reviews from CSV: {self.csv_path}")

    def get_bank_id(self, bank_code: str) -> Optional[int]:
        """Return bank_id from DB for a CSV bank code."""
        db_bank_name = self.bank_mapping.get(bank_code)
        if not db_bank_name:
            print(f"Skipping unknown bank: {bank_code}")
            return None
        self.cursor.execute("SELECT bank_id FROM banks WHERE bank_name = %s", (db_bank_name,))
        result = self.cursor.fetchone()
        if result is None:
            print(f"Bank not found in DB: {db_bank_name}")
            return None
        return result[0]

    @staticmethod
    def compute_sentiment_score(text: str) -> float:
        """Compute sentiment polarity as a score from -1 to 1."""
        return TextBlob(text).sentiment.polarity

    @staticmethod
    def process_theme(theme: str) -> str:
        """Ensure theme string is valid, default to 'Other'."""
        if pd.isna(theme) or theme.strip() == '':
            return "Other"
        return theme

    def insert_review_row(self, row: pd.Series) -> bool:
        """Insert a single review row into PostgreSQL."""
        bank_id = self.get_bank_id(row['bank'])
        if bank_id is None:
            return False

        sentiment_score = self.compute_sentiment_score(row['review_text'])
        theme = self.process_theme(row.get('theme', 'Other'))

        self.cursor.execute(
            """
            INSERT INTO reviews
            (bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, theme, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                bank_id,
                row['review_text'],
                row['rating'],
                row['date'],  # CSV 'date' → DB 'review_date'
                row['sentiment'],
                sentiment_score,
                theme,
                row.get('source', 'Google Play')
            )
        )
        return True

    def insert_all_reviews(self) -> None:
        """Insert all reviews from DataFrame into PostgreSQL."""
        if self.df is None:
            raise ValueError("CSV not loaded. Call load_csv() first.")

        inserted_count = 0
        for _, row in self.df.iterrows():
            if self.insert_review_row(row):
                inserted_count += 1

        self.conn.commit()
        print(f"{inserted_count} reviews inserted successfully into 'reviews' table.")

    def close_db(self) -> None:
        """Close PostgreSQL connection."""
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print(f"Database connection to {self.db_config.get('dbname')} closed.")

    def run(self) -> None:
        """Run the full insertion pipeline."""
        self.connect_db()
        self.load_csv()
        self.insert_all_reviews()
        self.close_db()


if __name__ == "__main__":
    db_config = {
        "host": "localhost",
        "port": "5432",
        "dbname": "bank_reviews",
        "user": "postgres",
        "password": "1234"
    }
    inserter = PostgresReviewInserter(
        db_config=db_config,
        csv_path="task-2/reviews_sentiment_themes.csv"
    )
    inserter.run()
