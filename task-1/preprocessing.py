# task-1/preprocessing.py

import pandas as pd
import re
from datetime import datetime
from typing import Optional


class ReviewPreprocessor:
    """
    A class to preprocess scraped bank app reviews.
    Includes text normalization, URL/emoji removal, date formatting, and deduplication.
    """

    def __init__(self, input_file: str, output_file: str):
        """
        Initialize the preprocessor with input and output paths.

        Args:
            input_file (str): Path to raw scraped reviews CSV.
            output_file (str): Path to save the cleaned reviews CSV.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """
        Load raw reviews from the CSV file.
        """
        self.df = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.df)} raw reviews from {self.input_file}")

    def clean_text(self) -> None:
        """
        Preprocess the review text:
        - Lowercase
        - Remove URLs
        - Remove emojis/special characters
        - Normalize whitespace
        - Drop missing or empty reviews
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Lowercase text
        self.df['review_text'] = self.df['review_text'].str.lower()

        # Remove URLs
        self.df['review_text'] = self.df['review_text'].apply(
            lambda x: re.sub(r'http\S+|www\S+', '', x)
        )

        # Remove emojis and special characters (keep letters, numbers, basic punctuation)
        self.df['review_text'] = self.df['review_text'].apply(
            lambda x: re.sub(r'[^\w\s.,!?]', '', x)
        )

        # Normalize whitespace
        self.df['review_text'] = self.df['review_text'].apply(
            lambda x: re.sub(r'\s+', ' ', x).strip()
        )

        # Drop missing or empty reviews
        self.df = self.df.dropna(subset=['review_text'])
        self.df = self.df[self.df['review_text'].str.strip() != '']

    def normalize_dates(self) -> None:
        """
        Normalize the date column to YYYY-MM-DD format.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.df['date'] = pd.to_datetime(self.df['date']).dt.strftime('%Y-%m-%d')

    def remove_duplicates(self) -> None:
        """
        Remove duplicate reviews based on review text and bank.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['review_text', 'bank'])
        after = len(self.df)
        print(f"Removed {before - after} duplicate reviews.")

    def save_cleaned_data(self) -> None:
        """
        Save the cleaned DataFrame to a CSV file.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.df.to_csv(self.output_file, index=False)
        print(f"Preprocessing complete. {len(self.df)} reviews saved to {self.output_file}")


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ReviewPreprocessor(
        input_file="data/raw_reviews.csv",
        output_file="task-1/clean_reviews.csv"
    )

    # Run preprocessing steps
    preprocessor.load_data()
    preprocessor.clean_text()
    preprocessor.normalize_dates()
    preprocessor.remove_duplicates()
    preprocessor.save_cleaned_data()
