from database import get_db, close_db, engine
from sqlalchemy import text
import pandas as pd
from sklearn.mixture import GaussianMixture
from feature_engineering_functions import export_functions
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def daily_fill(df, date_col='timestamp', value_col='value'):
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Check if DataFrame is empty or has no valid timestamps
    if df.empty or df[date_col].isna().all():
        return df
    
    df = df.sort_values(date_col).set_index(date_col)

    # Check if index has valid timestamps
    if df.index.isna().all():
        return df.reset_index().rename(columns={'index': date_col})

    full_range = pd.date_range(start=df.index.min(), end=datetime.utcnow().date(), freq='D')

    df_filled = df.reindex(full_range).ffill()

    df_filled = df_filled.reset_index().rename(columns={'index': date_col})
    
    return df_filled


def minute_15_fill(df, date_col='timestamp', value_col='value'):
    df[date_col] = pd.to_datetime(df[date_col])

    # Check if DataFrame is empty or has no valid timestamps
    if df.empty or df[date_col].isna().all():
        return df

    # Sort and set index
    df = df.sort_values(date_col).set_index(date_col)

    # Check if index has valid timestamps
    if df.index.isna().all():
        return df.reset_index().rename(columns={'index': date_col})

    # Create full 15-min range
    full_range = pd.date_range(
        start=df.index.min(),
        end=datetime.utcnow(),
        freq='15T'  # 15-minute frequency
    )

    # Reindex and forward fill
    df_filled = df.reindex(full_range).ffill()

    # Reset index and rename
    df_filled = df_filled.reset_index().rename(columns={'index': date_col})

    return df_filled


def build_dataset(dataset_id, model_type):
    db = get_db()
    
    # Get engineered features for this dataset
    query = text(f"SELECT * FROM dataset_row WHERE dataset_id = {dataset_id};")
    result = db.execute(query)
    dataset_rows = result.fetchall()

    # Get regime features for this dataset
    regime_query = text(f"SELECT * FROM dataset_row_regime WHERE dataset_id = {dataset_id};")
    regime_result = db.execute(regime_query)
    dataset_regime_rows = regime_result.fetchall()

    dataset_df = pd.DataFrame()

    cache = {}

    # Process engineered features
    for dataset_row in dataset_rows:
        engineered_feature_id = dataset_row[2]
        engineered_feature = db.execute(text(f"SELECT * FROM engineered_feature WHERE id = {engineered_feature_id};")).fetchone()

        engineered_feature_name = engineered_feature[2]
        feature_engineering_id = engineered_feature[1]
        feature_engineering = db.execute(text(f"SELECT * FROM feature_engineering WHERE id = {feature_engineering_id};")).fetchone()

        feature_engineering_id = feature_engineering[0]
        function_name = feature_engineering[1]

        feature_engineering_rows = db.execute(text(f"SELECT * FROM feature_engineering_row WHERE feature_engineering_id = {feature_engineering_id};")).fetchall()

        feature_engineering_df = pd.DataFrame()

        for feature_engineering_row in feature_engineering_rows:
            feature_id = feature_engineering_row[1]

            feature = db.execute(text(f"SELECT * FROM feature WHERE id = {feature_id};")).fetchone()

            try:
                df = cache[feature_id]
            except:
                result = db.execute(text(f"SELECT * FROM feature_row WHERE feature_id = {feature_id};"))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                df = df[["timestamp", "value"]]
                cache[feature_id] = df

            if feature_engineering_df.empty:
                feature_engineering_df["timestamp"] = df["timestamp"]

            try:
                base_name = feature[4].split("_")[2]
                column_name = base_name
                counter = 1

                while column_name in feature_engineering_df.columns:
                    column_name = f"{base_name}_{counter}"
                    counter += 1

                feature_engineering_df[column_name] = df["value"]
            except:
                feature_engineering_df["value"] = df["value"]

        feature_engineering_df[engineered_feature_name] = export_functions[function_name](feature_engineering_df)

        df = feature_engineering_df.dropna(subset=[engineered_feature_name])

        df = df[["timestamp", engineered_feature_name]]

        if model_type == "Master":
            df = daily_fill(df)
        elif model_type == "Intraday":
            df = minute_15_fill(df)

        if dataset_df.empty:
            dataset_df["timestamp"] = df["timestamp"]
            dataset_df[engineered_feature_name] = df[engineered_feature_name]
        else:
            dataset_df = pd.merge(
                dataset_df,
                df[["timestamp", engineered_feature_name]],
                on="timestamp",
                how="inner"
            )

    # Process regime features
    for dataset_regime_row in dataset_regime_rows:
        regime_model_id = dataset_regime_row[2]  # Assuming structure: (id, dataset_id, regime_model_id)
        
        # Calculate expected regime values
        regime_df = calculate_expected_regime(db, regime_model_id)
        
        if regime_df.empty:
            continue
        
        # Create feature name for this regime model
        regime_model = db.execute(text(f"SELECT model_name FROM regime_model WHERE id = {regime_model_id};")).fetchone()
        regime_feature_name = f"regime_{regime_model[0]}" if regime_model else f"regime_{regime_model_id}"
        
        # Apply daily fill to regime data - ensure proper column naming
        if model_type == "Master":
            regime_df_filled = daily_fill(regime_df, date_col='timestamp', value_col='expected_regime')
        elif model_type == "Intraday":
            regime_df_filled = minute_15_fill(regime_df, date_col='timestamp', value_col='expected_regime')
        regime_df_filled = regime_df_filled.rename(columns={'expected_regime': regime_feature_name})
        
        # Merge with main dataset
        if dataset_df.empty:
            dataset_df = regime_df_filled[["timestamp", regime_feature_name]].copy()
        else:
            dataset_df = pd.merge(
                dataset_df,
                regime_df_filled[["timestamp", regime_feature_name]],
                on="timestamp",
                how="inner"
            )

    close_db(db)

    return dataset_df


def calculate_expected_regime(db, regime_model_id):
    """
    Calculate expected regime value as sum(regime_name * probability) for each timestamp
    
    For each timestamp:
    - Get all regimes (0, 1, 2, ...) and their probabilities
    - Calculate: regime_0 * prob_0 + regime_1 * prob_1 + regime_2 * prob_2 + ...
    - Return single value per timestamp representing expected regime state
    """
    query = text("""
        SELECT
            rr.timestamp,
            r.name AS regime_name,
            rr.probability
        FROM
            regime_row rr
        JOIN
            regime r ON rr.regime_id = r.id
        WHERE
            r.regime_model_id = :regime_model_id
        ORDER BY rr.timestamp, r.name
    """)
    
    result = db.execute(query, {"regime_model_id": regime_model_id})
    df = pd.DataFrame(result.fetchall(), columns=["timestamp", "regime_name", "probability"])
    
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "expected_regime"])
    
    # Convert regime_name to numeric (regime names are stored as strings: "0", "1", "2", etc.)
    df['regime_name_numeric'] = pd.to_numeric(df['regime_name'], errors='coerce')
    
    # Remove any rows where regime_name couldn't be converted to numeric
    df = df.dropna(subset=['regime_name_numeric'])
    
    # Calculate weighted regime value: regime_name * probability
    df['weighted_regime'] = df['regime_name_numeric'] * df['probability']
    
    # Group by timestamp and sum the weighted values to get expected regime
    # This gives us: sum(regime_i * probability_i) for each timestamp
    expected_regime_df = df.groupby('timestamp')['weighted_regime'].sum().reset_index()
    expected_regime_df = expected_regime_df.rename(columns={'weighted_regime': 'expected_regime'})
    
    # Ensure timestamps are properly formatted
    expected_regime_df['timestamp'] = pd.to_datetime(expected_regime_df['timestamp'])
    
    
    return expected_regime_df


def upload_regimes(db, df, model_parameters, model_id):
    for i in range(int(model_parameters["n_components"])):
        result = db.execute(text(f"INSERT INTO regime (regime_model_id, name) VALUES ({model_id}, {i}) RETURNING id;"))
        regime_id = result.scalar()

        db.commit()

        regime_df = df[["timestamp", f"regime_{i}_prob"]]
        regime_df["regime_id"] = regime_id

        regime_df = regime_df.rename(columns={
            f"regime_{i}_prob": 'probability'
        })

        regime_df.to_sql("regime_row", con=engine, if_exists="append", index=False)

    db.commit()


def main(specific_models):
    db = get_db()
    
    special_job = True

    if len(specific_models) == 0:
        special_job = False

    for model_objs in db.execute(text(f"SELECT * FROM regime_model;")).fetchall():
        model_id = model_objs[0]

        if special_job and model_id not in specific_models:
            continue

        model = db.execute(text(f"SELECT * FROM regime_model WHERE id = {model_id}")).fetchone()

        dataset_id = model[1]
        model_type = model[2]
        model_parameters = model[3]

        dataset_df = build_dataset(dataset_id, model_type)

        X = dataset_df.drop(columns=['timestamp'])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GaussianMixture(n_components=model_parameters["n_components"], random_state=model_parameters["random_state"])
        model.fit(X_scaled)

        regimes = model.predict(X_scaled)
        regime_list = model.predict_proba(X_scaled)

        df = X.copy()
        df['regime'] = regimes

        regime_dict = {
            "timestamp": dataset_df['timestamp']
        }
        for i in range(regime_list.shape[1]):
            regime_dict[f"regime_{i}_prob"] = regime_list[:, i]

        df = pd.DataFrame(regime_dict)

        regimes = db.execute(text(f"SELECT * FROM regime WHERE regime_model_id = {model_id};")).fetchall()

        if len(regimes) != 0:
            for regime in regimes:
                db.execute(text(f"DELETE FROM regime_row WHERE regime_id = {regime[0]}"))
                db.execute(text(f"DELETE FROM regime WHERE id = {regime[0]}"))

                db.commit()
        
        upload_regimes(db, df, model_parameters, model_id)
            
    close_db(db)