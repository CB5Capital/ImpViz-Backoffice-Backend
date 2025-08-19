import pandas as pd
from database import get_db, close_db, insert_into_table, engine
from datetime import datetime
import json
from dotenv import load_dotenv
import os

load_dotenv()

def databento_historical_data(df, target):
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df['date'] = df['ts_event'].dt.date

    df_sorted = df.sort_values(by=['date', 'volume'], ascending=[True, False])

    daily_top = df_sorted.groupby('date').first().reset_index()

    result = daily_top[['date', target]]

    result = result.rename(columns={
        'date': 'timestamp',
        target: 'value'
    })

    return result

def databento_historical_data_minute(df, target):
    # Ensure datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'])

    # Bucket to minute
    minute = df['ts_event'].dt.floor('T')

    # Pick the row with max volume within each minute
    idx = df.groupby(minute)['volume'].idxmax()

    # Build result
    picked = df.loc[idx].copy().sort_values('ts_event')
    result = picked[['ts_event', target]].rename(columns={'ts_event': 'timestamp', target: 'value'}).reset_index(drop=True)
    return result

def tradingview_historical_data(df, target):
    if target == "volume":
        target = "Volume"

    df['time'] = pd.to_datetime(df['time'])

    df['date'] = df['time'].dt.date

    result = df.rename(columns={
        'date': 'timestamp',
        target: 'value'
    })

    return result[["timestamp", "value"]]

def upload_pandas_dataframe(df, feature_id, db_session):
    df["feature_id"] = feature_id

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.to_sql("feature_row", con=db_session.connection(), if_exists="append", index=False)

def fred_historical_data(df, target):
    df['observation_date'] = pd.to_datetime(df['observation_date'])

    result = df.rename(columns={
        'observation_date': 'timestamp',
        target: 'value'
    })

    return result.dropna()

def market_data(df, data_structure, category_id, sub_category, source, frequency, unit, plain_name, source_symbol, description, db_session):
    org_df = df

    if source == "FRED":
        feature_id = insert_into_table(db_session, "feature", {
            "category_id" : category_id,
            "sub_category" : sub_category,
            "source" : source,
            "frequency" : frequency,
            "unit" : unit,
            "name" : plain_name,
            "description" : description
        }, True)

        df = fred_historical_data(org_df, source_symbol)

        insert_into_table(db_session, "feature_updater", {
            "feature_id" : feature_id,
            "api_config" : json.dumps({
                "request_params" : f"?series_id={source_symbol}&api_key={os.getenv('FRED_API_KEY')}&file_type=json"
            })
        }, False)

        upload_pandas_dataframe(df, feature_id, db_session)
        return

    if data_structure == "OHLCV":
        targets = ["close", "open", "high", "low", "volume"]
    elif data_structure == "close_only":
        targets = ["close"]
    
    for target in targets:
        name = plain_name+f" ({target})"
        target_source = source+"_"+target

        if target != "volume":
            description = f"{target} price for {plain_name}"
        else:
            description = f"{target} for {plain_name}"
        
        feature_id = insert_into_table(db_session, "feature", {
            "category_id" : category_id,
            "sub_category" : sub_category,
            "source" : target_source,
            "frequency" : frequency,
            "unit" : unit,
            "name" : name,
            "description" : description
        }, True)

        if source.split("_")[0] == "Databento":
            if frequency == "Daily":
                df = databento_historical_data(org_df, target)
            elif frequency == "1Minute":
                df = databento_historical_data_minute(org_df, target)
        elif source.split("_")[0] == "TradingView":
            df = tradingview_historical_data(org_df, target)

        if source.split("_")[0] == "Databento":
            if frequency == "Daily":
                insert_into_table(db_session, "feature_updater", {
                    "feature_id" : feature_id,
                    "api_config" : json.dumps({
                        "dataset" : "GLBX.MDP3",
                        "schema" : "OHLCV-1d",
                        "symbols" : [source_symbol],
                        "target_column": target
                    })
                }, False)
            elif frequency == "1Minute":
                insert_into_table(db_session, "feature_updater", {
                    "feature_id" : feature_id,
                    "api_config" : json.dumps({
                        "dataset" : "GLBX.MDP3",
                        "schema" : "OHLCV-1m",
                        "symbols" : [source_symbol],
                        "target_column": target
                    })
                }, False)

        upload_pandas_dataframe(df, feature_id, db_session)

def macro_data(df, data_structure, category_id, sub_category, source, frequency, unit, plain_name, source_symbol, description, db_session):
    feature_id = insert_into_table(db_session, "feature", {
        "category_id" : category_id,
        "sub_category" : sub_category,
        "source" : source,
        "frequency" : frequency,
        "unit" : unit,
        "name" : plain_name,
        "description" : description
    }, True)

    df = fred_historical_data(df, source_symbol)

    insert_into_table(db_session, "feature_updater", {
        "feature_id" : feature_id,
        "api_config" : json.dumps({
            "request_params" : f"?series_id={source_symbol}&api_key={os.getenv('FRED_API_KEY')}&file_type=json"
        })
    }, False)

    upload_pandas_dataframe(df, feature_id, db_session)