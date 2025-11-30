from google_play_scraper import reviews, Sort
import pandas as pd
from typing import List, Dict


class BankReviewScraper:
    """
    A class to scrape app reviews from Google Play for multiple banks
    and save the results to a CSV file.
    """

    def __init__(self, apps: Dict[str, str], reviews_per_app: int = 400):
        """
        Initializes the scraper with app IDs and review count.

        Args:
            apps (Dict[str, str]): A dictionary mapping bank names to their Google Play app IDs.
            reviews_per_app (int, optional): Number of reviews to fetch per app. Defaults to 400.
        """
        self.apps = apps
        self.reviews_per_app = reviews_per_app
        self.all_reviews: List[Dict] = []

    def scrape_reviews_for_app(self, bank: str, app_id: str) -> None:
        """
        Scrape reviews for a single app and append them to the all_reviews list.

        Args:
            bank (str): Name of the bank.
            app_id (str): Google Play app ID.
        """
        print(f"Scraping reviews for {bank}...")

        # Fetch reviews using google_play_scraper
        result, _ = reviews(
            app_id,
            lang='en',
            country='ET',
            sort=Sort.NEWEST,
            count=self.reviews_per_app
        )

        # Append each review as a dictionary
        for r in result:
            self.all_reviews.append({
                "review_text": r['content'],
                "rating": r['score'],
                "date": r['at'].strftime("%Y-%m-%d"),
                "bank": bank,
                "source": "Google Play"
            })

    def scrape_all_reviews(self) -> None:
        """
        Scrape reviews for all apps defined in the apps dictionary.
        """
        for bank, app_id in self.apps.items():
            self.scrape_reviews_for_app(bank, app_id)

    def save_to_csv(self, file_path: str) -> None:
        """
        Save all scraped reviews to a CSV file.

        Args:
            file_path (str): Path to save the CSV file.
        """
        df = pd.DataFrame(self.all_reviews)
        df.to_csv(file_path, index=False)
        print(f"Scraping complete! Saved to {file_path}")


if __name__ == "__main__":
    # Define banks and their Google Play app IDs
    apps = {
        "CBE": "com.combanketh.mobilebanking",
        "BOA": "com.boa.boaMobileBanking",
        "Dashen": "com.dashen.dashensuperapp"
    }

    # Initialize scraper
    scraper = BankReviewScraper(apps=apps, reviews_per_app=400)

    # Scrape reviews for all apps
    scraper.scrape_all_reviews()

    # Save scraped reviews to CSV
    scraper.save_to_csv("data/raw_reviews.csv")
