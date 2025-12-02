# Ethiopian Bank App Reviews – Week 2 Challenge

This repository contains scripts for scraping, preprocessing, sentiment and theme analysis, and database engineering for three Ethiopian banks’ mobile apps:

* **Commercial Bank of Ethiopia (CBE)**
* **Bank of Abyssinia (BOA)**
* **Dashen Bank**

The goal is to analyze customer satisfaction, identify app usage pain points, and provide actionable insights for improvement.

---

## Repository Structure

| Folder         | Description |
|----------------|-------------|
| `task-1/`      | Scraping and preprocessing scripts |
| `task-2/`      | Sentiment and thematic analysis scripts |
| `task-3/`      | PostgreSQL database insertion scripts |
| `data/`        | Raw and cleaned datasets |

---

## Requirements

Install dependencies via `requirements.txt`. Key packages include:

* `pandas`
* `google-play-scraper`
* `textblob`
* `scikit-learn`
* `psycopg2-binary`

---

## Project Workflow

The analysis follows five main stages:

1. **Scraping (Task 1)** – Extract reviews, ratings, and dates from Google Play Store.
2. **Preprocessing (Task 1)** – Clean data, remove duplicates, normalize dates, and structure CSV.
3. **Sentiment & Theme Analysis (Task 2)** – Compute sentiment polarity, classify reviews, and extract recurring themes per bank.
4. **Database Insertion (Task 3)** – Store cleaned and enriched review data in PostgreSQL for structured access and querying.
5. **Insights & Recommendations (Task 4)** – Summarize results with visualizations, identify drivers and pain points, and recommend app improvements.

---

## Task 1: Data Collection and Preprocessing

* **Scripts:** `task-1/playstore_scraper.py`, `task-1/preprocessing.py`
* **Steps Completed:**
  - Scrape ~400 reviews per bank (~1,200 total)
  - Remove duplicates and handle missing data
  - Normalize dates (YYYY-MM-DD)
  - Output CSV: `task-1/clean_reviews.csv`
* **KPIs Achieved:** 1,200+ clean reviews collected, Git repo organized

---

## Task 2: Sentiment Analysis & Thematic Extraction

* **Script:** `task-2/sentiment_analysis.py`
* **Pipeline:**
  1. Load preprocessed CSV
  2. Compute sentiment polarity via TextBlob
  3. Classify as `positive`, `neutral`, or `negative`
  4. Extract top keywords using TF-IDF and group into themes
* **Output:** `task-2/reviews_sentiment_themes.csv`
* **KPIs Achieved:** All reviews scored, multiple themes assigned, modular pipeline ready

---

## Task 3: PostgreSQL Database Insertion

* **Script:** `task-3/insert_reviews_postgres.py`
* **Database:** `bank_reviews`
* **Tables:**
  - `banks` (bank_id, bank_name, app_name)
  - `reviews` (review_id, bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, theme, source)
* **Implementation:**
  - Map CSV bank codes to DB IDs
  - Insert multiple themes per review
* **KPIs Achieved:** >1,000 reviews inserted, schema documented

---

## Task 4: Insights & Recommendations (Next Steps)

* Visualize sentiment and theme trends (Matplotlib/Seaborn)
* Identify satisfaction drivers and pain points per bank
* Recommend app improvements
* Generate final report with plots and actionable insights

---

## Status

| Task                                | Status        |
| ----------------------------------- | ------------- |
| Task 1 – Scraping & Preprocessing   | ✅ Completed   |
| Task 2 – Sentiment & Themes         | ✅ Completed   |
| Task 3 – PostgreSQL DB              | ✅ Completed   |
| Task 4 – Insights & Recommendations | ✅ Completed |

---

## References

* [google-play-scraper](https://github.com/JoMingyu/google-play-scraper)  
* [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)  
* [TextBlob](https://textblob.readthedocs.io/en/dev/)  
* [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)  

---

**Author:** Mikiyas Dawit  
**Date:** December 2, 2025
