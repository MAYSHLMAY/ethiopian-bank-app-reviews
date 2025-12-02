# PostgreSQL Review Inserter

## Overview

`insert_reviews_postgres.py` is a Python script designed to insert preprocessed bank app reviews from a CSV file into a **PostgreSQL** database. It follows an **object-oriented and modular** design, making it reusable and easy to maintain.

The script handles:

* Database connection to PostgreSQL via `psycopg2`
* Loading cleaned review data from CSV (`reviews_sentiment_themes.csv`)
* Mapping CSV bank codes to database `banks` table
* Computing sentiment scores using TextBlob
* Handling multiple themes (defaulting to "Other" if missing)
* Inserting reviews into the `reviews` table

---

## Prerequisites

1. **PostgreSQL server installed** (tested with PostgreSQL 15.15)
2. Python 3.9+ environment
3. Python dependencies:

```bash
pip install psycopg2-binary pandas textblob
```

4. PostgreSQL database and tables created:

```sql
-- Banks table
CREATE TABLE IF NOT EXISTS banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name TEXT NOT NULL,
    app_name TEXT NOT NULL
);

-- Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INT REFERENCES banks(bank_id),
    review_text TEXT,
    rating INT,
    review_date DATE,
    sentiment_label TEXT,
    sentiment_score FLOAT,
    theme TEXT,
    source TEXT
);
```

5. CSV file (`reviews_sentiment_themes.csv`) containing preprocessed reviews, with at least the following columns:

```
bank, review_text, rating, date, sentiment, theme, source
```

---

## Usage

1. Update PostgreSQL connection parameters in `db_config`:

```python
db_config = {
    "host": "localhost",
    "port": "5432",
    "dbname": "bank_reviews",
    "user": "postgres",
    "password": "YOUR_PASSWORD"
}
```

2. Ensure your CSV path is correct:

```python
csv_path="task-2/reviews_sentiment_themes.csv"
```

3. Run the script:

```bash
python task-3/insert_reviews_postgres.py
```

---

## Class: `PostgresReviewInserter`

### Attributes

* `db_config`: Dictionary containing PostgreSQL connection parameters.
* `csv_path`: Path to CSV file containing reviews.
* `bank_mapping`: Optional dictionary mapping CSV bank codes to database bank names.

### Methods

* `connect_db()`: Connects to PostgreSQL.
* `load_csv()`: Loads the CSV into a pandas DataFrame.
* `get_bank_id(bank_code)`: Returns `bank_id` from DB for a CSV bank code.
* `compute_sentiment_score(text)`: Computes sentiment polarity (-1 to 1) using TextBlob.
* `process_theme(theme)`: Ensures theme string is valid, defaults to "Other".
* `insert_review_row(row)`: Inserts a single review row into PostgreSQL.
* `insert_all_reviews()`: Inserts all reviews from DataFrame.
* `close_db()`: Closes PostgreSQL connection.
* `run()`: Runs the full insertion pipeline (connect → load CSV → insert → close).

---

## Output

* Inserts all valid reviews into the `reviews` table in PostgreSQL.
* Prints the number of reviews successfully inserted.
* Handles missing or unknown banks gracefully.

---

## Notes

* Ensure the `banks` table is populated with the banks present in the CSV before running the script.
* The script automatically computes sentiment scores from review text using TextBlob.
* Themes missing in the CSV are automatically assigned `"Other"`.

---