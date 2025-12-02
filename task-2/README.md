# Task 2 – Sentiment & Thematic Analysis

This module performs sentiment analysis and thematic keyword extraction on Google Play Store reviews for three Ethiopian fintech banking apps.

## Objectives
- Classify each review as **positive**, **negative**, or **neutral**.
- Compute a **sentiment polarity score**.
- Extract high-impact keywords using **TF-IDF**.
- Assign each review to a **theme category** (e.g., “Account Access Issues”).

No visualizations are required in Task 2 (plots belong to Task 4).

---

## Folder Structure

```

task-2/
│
├── sentiment/
│   ├── sentiment_analyzer.py
│   ├── theme_extractor.py
│   └── **init**.py
│
├── run_task2_pipeline.py
├── data/
│   ├── reviews_cleaned.csv
│   └── reviews_sentiment_themes.csv
└── README.md

````

---

## Components

### 1. `SentimentAnalyzer`
- Uses `distilbert-base-uncased-finetuned-sst-2-english`
- Outputs:
  - `sentiment_label`
  - `sentiment_score`

### 2. `ThemeExtractor`
- Extracts top TF-IDF keywords  
- Maps each review to a theme using rule-based matching

### 3. `Task2Pipeline`
Orchestrates:
1. Data loading  
2. Sentiment classification  
3. Keyword extraction  
4. Theme assignment  
5. Saving final CSV  

---

## Usage

```bash
python run_task2_pipeline.py
````

---

## Output File

`reviews_sentiment_themes.csv`:

| review_text          | sentiment_label | sentiment_score | keywords              | theme              |
| -------------------- | --------------- | --------------- | --------------------- | ------------------ |
| "App crashes always" | NEGATIVE        | 0.98            | ["crash","error",...] | App Crashes & Bugs |

---

## No Graphs in Task 2

Per challenge instructions, **visualizations are part of Task 4**, not Task 2.

---
