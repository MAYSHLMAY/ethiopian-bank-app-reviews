# Ethiopian Bank App Reviews – Week 2 Challenge

This repository contains scraping, preprocessing, sentiment analysis, and database engineering tasks for three Ethiopian banks’ mobile apps:

* Commercial Bank of Ethiopia (CBE)
* Bank of Abyssinia (BOA)
* Dashen Bank

The goal is to analyze customer satisfaction and identify key drivers and pain points in app usage.

---

## Folders

* **task-1/**: Scraping and preprocessing scripts
* **task-2/**: Sentiment and thematic analysis
* **task-3/**: PostgreSQL database insertion scripts
* **data/**: Raw and cleaned datasets

---

## Requirements

See `requirements.txt` for required Python packages. Key dependencies include:

* `pandas`
* `google-play-scraper`
* `textblob`
* `scikit-learn`
* `psycopg2-binary`

---

## Task 1: Data Collection and Preprocessing

### Overview

Scrape and clean user reviews from the Google Play Store for three Ethiopian banking apps.

### Steps Completed

1. **Scraping**

   * Extracted reviews, ratings, dates, and app names
   * Minimum ~400 reviews per bank (~1,200 total)
   * Script: `task-1/playstore_scraper.py`
2. **Preprocessing**

   * Removed duplicates
   * Handled missing data
   * Normalized dates (YYYY-MM-DD)
   * Saved clean dataset as CSV: `task-1/clean_reviews.csv`
   * Script: `task-1/preprocessing.py`

### KPIs Achieved

* 1,200+ clean reviews collected
* Organized Git repo with meaningful commits

---

## Task 2: Sentiment Analysis and Thematic Extraction

### Overview

Compute sentiment for each review and extract themes to identify satisfaction drivers and pain points.

### Pipeline

1. **Load Preprocessed Data**

   * Input: `task-1/clean_reviews.csv`
   * Columns: `review_text`, `rating`, `date`, `bank`, `source`

2. **Sentiment Analysis**

   * Method: TextBlob polarity scores
   * Classification: `positive`, `negative`, `neutral`
   * Output: `sentiment` column

3. **Keyword & Theme Extraction**

   * Extracted top recurring keywords per bank using TF-IDF
   * Grouped keywords into multiple themes per bank
   * Output CSV: `task-2/reviews_sentiment_themes.csv`

### Usage

```bash
python task-2/sentiment_analysis.py
```

### KPIs Achieved

* Sentiment scores computed for all reviews
* Multiple themes assigned per bank
* Modular Python pipeline committed to GitHub

---

## Task 3: Store Cleaned Data in PostgreSQL

### Overview

Persistently store cleaned and processed review data in PostgreSQL.

### Implementation

* PostgreSQL database: `bank_reviews`
* Tables created:

  * `banks` (bank_id, bank_name, app_name)
  * `reviews` (review_id, bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, theme, source)
* OOP Python insertion script: `task-3/insert_reviews_postgres.py`

  * Loads CSV review data
  * Maps CSV bank codes to DB IDs
  * Computes sentiment scores
  * Inserts multiple themes into `reviews` table

### KPIs Achieved

* PostgreSQL database connection verified
* > 1,000 reviews inserted into `reviews` table
* Schema documented and committed to GitHub

---

## Status (Current)

| Task                                | Status        |
| ----------------------------------- | ------------- |
| Task 1 – Scraping & Preprocessing   | ✅ Completed   |
| Task 2 – Sentiment & Themes         | ✅ Completed   |
| Task 3 – PostgreSQL DB              | ✅ Completed   |
| Task 4 – Insights & Recommendations | ⬜ Not started |

---

## Next Steps (Task 4)

* Derive actionable insights from sentiment and themes
* Visualize key trends (Matplotlib, Seaborn)
* Identify drivers (e.g., fast navigation) and pain points (e.g., crashes) per bank
* Recommend app improvements based on analysis
* Prepare final report with plots and recommendations

---

## References

* [google-play-scraper](https://github.com/JoMingyu/google-play-scraper)
* [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
* [TextBlob](https://textblob.readthedocs.io/en/dev/)
* [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

---

**Author:** Mikiyas Dawit
**Date:** December 2, 2025

---
