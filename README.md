Perfect! Here’s a **full, updated README.md** for your **interim submission**, reflecting everything you’ve done for Task 1 and Task 2:

````markdown
# Ethiopian Bank App Reviews – Week 2 Challenge

This repository contains scraping, preprocessing, sentiment analysis, and database engineering tasks for three Ethiopian banks’ mobile apps:

- Commercial Bank of Ethiopia (CBE)
- Bank of Abyssinia (BOA)
- Dashen Bank

The goal is to analyze customer satisfaction and identify key drivers and pain points in app usage.

---

## Folders

- **task-1/**: Scraping and preprocessing scripts  
- **task-2/**: Sentiment and thematic analysis  
- **data/**: Raw and cleaned datasets

---

## Requirements

See `requirements.txt` for required Python packages. Key dependencies include:

- `pandas`  
- `google-play-scraper`  
- `textblob`  
- `scikit-learn`  

---

## Task 1: Data Collection and Preprocessing

### Overview
Scrape and clean user reviews from the Google Play Store for three Ethiopian banking apps.

### Steps Completed
1. **Scraping**
   - Extracted reviews, ratings, dates, and app names
   - Minimum ~400 reviews per bank (~1,200 total)
   - Script: `task-1/playstore_scraper.py`
   
2. **Preprocessing**
   - Remove duplicates
   - Handle missing data
   - Normalize dates (YYYY-MM-DD)
   - Save clean dataset as CSV: `task-1/clean_reviews.csv`
   - Script: `task-1/preprocessing.py`

### KPIs Achieved
- 1,200+ reviews collected  
- Clean CSV dataset ready for analysis  
- Organized Git repo with meaningful commits  

---

## Task 2: Sentiment Analysis and Thematic Extraction

### Overview
Compute sentiment for each review and extract preliminary themes to identify drivers and pain points.

### Pipeline
1. **Load Preprocessed Data**
   - Input: `task-1/clean_reviews.csv`
   - Columns: `review_text`, `rating`, `date`, `bank`, `source`

2. **Sentiment Analysis**
   - Method: TextBlob polarity scores
   - Classification: `positive`, `negative`, `neutral`
   - Output: New column `sentiment` added to the dataframe

3. **Keyword & Theme Extraction (Interim)**
   - Top recurring keywords extracted per bank using TF-IDF
   - Preliminary grouping into 2–3 themes per bank:
     - Example: *Login Issues*, *App Performance*, *Customer Support*
   - Output CSV: `task-2/reviews_sentiment_themes.csv`  
     Columns: `review_text`, `rating`, `date`, `bank`, `source`, `sentiment`, `theme(s)`

### Usage
Run the sentiment analysis script:

```bash
python task-2/sentiment_analysis.py
````

This generates a CSV with sentiment labels ready for further visualization and insights.

### KPIs Achieved

* Sentiment scores computed for all scraped reviews
* Preliminary keyword extraction and theme grouping completed
* Modular Python pipeline committed to GitHub

---

## Status (Interim Submission)

| Task                                | Status              |
| ----------------------------------- | ------------------- |
| Task 1 – Scraping & Preprocessing   | ✅ Completed         |
| Task 2 – Sentiment & Themes         | ⚠ Partial (interim) |
| Task 3 – PostgreSQL DB              | ⬜ Not started       |
| Task 4 – Insights & Recommendations | ⬜ Not started       |

---

## Next Steps

* Complete Task 2 with full thematic analysis
* Begin Task 3: Store cleaned data in PostgreSQL
* Prepare preliminary visualizations for interim report
* Continue modular commits for traceable Git history

---

## References

* [google-play-scraper](https://github.com/JoMingyu/google-play-scraper)
* [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
* [TextBlob](https://textblob.readthedocs.io/en/dev/)
* [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

---

**Author:** Mikiyas Dawit
**Submission:** Week 2 Interim Report – Omega Consultancy Challenge
**Date:** December 1, 2025

```
