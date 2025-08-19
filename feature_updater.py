from database import engine, get_db, get_table_data, close_db
from sqlalchemy import text
import requests
import databento
from datetime import datetime, timedelta
from feature_engineering_functions import export_functions
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

def update_feature_fred(feature_id):
    db = get_db()

    query = text(f"SELECT * FROM feature_updater WHERE feature_id = {feature_id};")

    result = db.execute(query)
    
    updater = result.fetchone()

    update_config = updater[2]

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = update_config["request_params"]

    response = requests.get(url+params)

    feature_data = response.json()["observations"]

    timestamp = feature_data[-1]["date"]
    value = feature_data[-1]["value"]

    query = text("""
        INSERT INTO feature_row (feature_id, timestamp, value)
        VALUES (:feature_id, :timestamp, :value)
        ON CONFLICT (feature_id, timestamp)
        DO UPDATE SET value = EXCLUDED.value
    """)

    db.execute(query, {
        "feature_id": feature_id,
        "timestamp": timestamp,
        "value": float(value)
    })
    db.commit()

    close_db(db)
    
def update_feature_databento(feature_id):
    db = get_db()

    query = text(f"SELECT * FROM feature_updater WHERE feature_id = {feature_id};")

    result = db.execute(query)
    
    updater = result.fetchone()

    update_config = updater[2]
    
    client = databento.Historical(os.getenv('DATABENTO_API_KEY'))

    today = datetime.today()
    if today.weekday() == 6:
        today = today - timedelta(days=1)

    query = text(f"SELECT * FROM feature_row WHERE feature_id = {feature_id} ORDER BY timestamp DESC LIMIT 1;")

    result = db.execute(query)

    latest_row_date = result.fetchone()[3]

    today_str = today.date().strftime('%Y-%m-%dT%H:%M:%S')
    days_ago_str = latest_row_date.strftime('%Y-%m-%dT%H:%M:%S')

    limit = today - latest_row_date

    data = client.timeseries.get_range(
        dataset=update_config["dataset"],
        symbols=update_config["symbols"],
        schema=update_config["schema"],
        start=days_ago_str,
        end=today_str,
        limit=limit.days
    )

    df = data.to_df()

    timestamp = pd.to_datetime(df.index[-1])
    value = df[update_config["target_column"]].iloc[-1]

    query = text("""
        INSERT INTO feature_row (feature_id, timestamp, value)
        VALUES (:feature_id, :timestamp, :value)
        ON CONFLICT (feature_id, timestamp)
        DO UPDATE SET value = EXCLUDED.value
    """)

    db.execute(query, {
        "feature_id": feature_id,
        "timestamp": timestamp,
        "value": float(value)
    })
    db.commit()

    close_db(db)

def main():
    db = get_db()

    query = text("SELECT * FROM feature;")

    result = db.execute(query)

    features = result.fetchall()

    close_db(db)

    for feature in features:
        feature_id = feature[0]
        frequency = feature[6]

        try:
            source = feature[4].split("_")[0]
        except:
            source = feature[4]

        if source == "Databento":
            if frequency != "Daily":
                continue
            update_feature_databento(feature_id)
        elif source == "FRED":
            update_feature_fred(feature_id)
        elif source in ["TradingView", "Investing.com"]:
            pass  # Features need to be updated manually