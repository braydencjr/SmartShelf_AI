# run_anomaly_to_db.py

import os
import pandas as pd
import pymysql
from dotenv import load_dotenv
from anomaly_detection import detect_anomalies_zscore, detect_anomalies_isolation_forest

# --- Load environment variables ---
load_dotenv()
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")

# --- Connect to MySQL ---
conn = pymysql.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DB,
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor,
    ssl_disabled=True
)

# --- Load CSV ---
csv_path = "data/daily_sales.csv"  # Change to your uploaded CSV path
daily_df = pd.read_csv(csv_path, parse_dates=["sale_date"])

# --- Detect anomalies ---
zscore_anomalies = detect_anomalies_zscore(daily_df)
iso_anomalies = detect_anomalies_isolation_forest(daily_df)

# Add a column to identify model
zscore_anomalies['model_detected'] = 'ZScore'
iso_anomalies['model_detected'] = 'IsolationForest'

# Combine anomalies
all_anomalies = pd.concat([zscore_anomalies, iso_anomalies], ignore_index=True)

# --- Insert anomalies into database ---
with conn.cursor() as cursor:
    for _, row in all_anomalies.iterrows():
        cursor.execute("""
            INSERT INTO anomalies (sale_date, product_id, daily_sales, anomaly_type, z_score, model_detected)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                anomaly_type = VALUES(anomaly_type),
                z_score = VALUES(z_score),
                model_detected = VALUES(model_detected)
        """, (
            row.get("sale_date"),
            row.get("product_id"),
            row.get("daily_sales"),
            row.get("anomaly_type"),
            row.get("z_score") if "z_score" in row else None,
            row.get("model_detected")
        ))
    conn.commit()

conn.close()
print("✅ Anomalies detected and inserted into database successfully!")
