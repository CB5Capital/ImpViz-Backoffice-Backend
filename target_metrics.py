from database import get_db, close_db, insert_into_table, engine, get_session
from sqlalchemy import text
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)

function_metafield_input_lookup = {
    "hit_rate" : ["P&L"],
    "pnl_per_day" : ["P&L"],
    "P&L_per_direction" : ["P&L", "Direction"],
    "total_pnl_per_day_per_direction" : ["P&L", "Direction"],
    "profitable_trades_drawdown_per_direction" : ["P&L", "Drawdown", "Direction"],
    "hit_rate_per_direction" : ["P&L", "Direction"],
    "pnl_split_wins_losses" : ["P&L"],
    "average_duration_by_type" : ["P&L", "Direction", "Duration in Minutes"],
    "drawdown_per_direction" : ["P&L", "Drawdown", "Direction"],
    "pnl_per_trade_per_direction_per_hour" : ["P&L", "Direction"]
}

def calculate_12_month_return(filename, target_metric_id, db_session=None):
    df = pd.read_csv(filename)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["12m_forward_return"] = (df["close"].shift(-252) / df["close"] - 1) * 100

    df = df.dropna(subset=["12m_forward_return"]).reset_index(drop=True)

    df = pd.DataFrame({
        "timestamp": df["time"],
        "value": df["12m_forward_return"]
    })

    df["target_metric_id"] = target_metric_id
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Use the provided session connection or fall back to engine
    if db_session:
        # Use session's connection for bulk insert while maintaining transaction
        df.to_sql("target_metric_row", con=db_session.connection(), if_exists="append", index=False)
    else:
        # Fallback to engine (for backward compatibility)
        df.to_sql("target_metric_row", con=db_session.connection(), if_exists="append", index=False)

def run_backtest_function(db_session, function, target_metric_id, metadata_names):
    def raw(df, target_metric_id, metadata_name):
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        df["date"] = df["timestamp"].dt.date

        grouped = df.groupby("date")["value"].mean().reset_index()
        grouped.rename(columns={"date": "timestamp"}, inplace=True)

        grouped["name"] = metadata_name
        grouped["target_metric_id"] = target_metric_id

        grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    def hit_rate(df, target_metric_id, metadata_name):
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["hit_rate"] = ((df["value"] > 0).astype(int))*100
        df["date"] = df["timestamp"].dt.date

        grouped = df.groupby("date")["hit_rate"].mean().reset_index()
        grouped.rename(columns={"date": "timestamp", "hit_rate": "value"}, inplace=True)

        grouped["name"] = "Hit rate"
        grouped["target_metric_id"] = target_metric_id

        grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    def pnl_per_day(df, target_metric_id, metadata_name):
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        df["date"] = df["timestamp"].dt.date

        grouped = df.groupby("date")["value"].sum().reset_index()
        grouped.rename(columns={"date": "timestamp"}, inplace=True)

        grouped["name"] = "Total P&L per day"
        grouped["target_metric_id"] = target_metric_id

        grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    def pnl_per_direction(df, target_metric_id, metadata_name):
        pivoted_df = df.pivot(index='timestamp', columns='name', values='value').reset_index()

        pivoted_df['P&L'] = pd.to_numeric(pivoted_df['P&L'], errors='coerce')

        long_df = pivoted_df[pivoted_df['Direction'] == 'Long'][['timestamp', 'P&L']].rename(columns={'P&L': 'value'})
        short_df = pivoted_df[pivoted_df['Direction'] == 'Short'][['timestamp', 'P&L']].rename(columns={'P&L': 'value'})

        long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
        long_df["date"] = long_df["timestamp"].dt.date

        grouped = long_df.groupby("date")["value"].mean().reset_index()
        grouped.rename(columns={"date": "timestamp"}, inplace=True)

        grouped["name"] = "P&L, Long trades only"
        grouped["target_metric_id"] = target_metric_id

        grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

        short_df["timestamp"] = pd.to_datetime(short_df["timestamp"])
        short_df["date"] = short_df["timestamp"].dt.date

        grouped = short_df.groupby("date")["value"].mean().reset_index()
        grouped.rename(columns={"date": "timestamp"}, inplace=True)

        grouped["name"] = "P&L, Short trades only"
        grouped["target_metric_id"] = target_metric_id

        grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    def total_pnl_per_day_per_direction(df, target_metric_id, metadata_name):
        """
        Calculate total P&L per day, separated by direction (Long/Short)
        """
        # Pivot the data to get P&L and Direction as separate columns
        pivoted_df = df.pivot(index='timestamp', columns='name', values='value').reset_index()
        
        # Ensure P&L is numeric
        pivoted_df['P&L'] = pd.to_numeric(pivoted_df['P&L'], errors='coerce')
        pivoted_df["timestamp"] = pd.to_datetime(pivoted_df["timestamp"])
        pivoted_df["date"] = pivoted_df["timestamp"].dt.date
        
        # Process Long trades
        long_df = pivoted_df[pivoted_df['Direction'] == 'Long'].copy()
        if not long_df.empty:
            long_grouped = long_df.groupby("date")["P&L"].sum().reset_index()
            long_grouped.rename(columns={"date": "timestamp", "P&L": "value"}, inplace=True)
            long_grouped["name"] = "Total P&L per day - Long"
            long_grouped["target_metric_id"] = target_metric_id
            
            long_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
        
        # Process Short trades
        short_df = pivoted_df[pivoted_df['Direction'] == 'Short'].copy()
        if not short_df.empty:
            short_grouped = short_df.groupby("date")["P&L"].sum().reset_index()
            short_grouped.rename(columns={"date": "timestamp", "P&L": "value"}, inplace=True)
            short_grouped["name"] = "Total P&L per day - Short"
            short_grouped["target_metric_id"] = target_metric_id
            
            short_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
        
        # Process Combined (all directions)
        combined_grouped = pivoted_df.groupby("date")["P&L"].sum().reset_index()
        combined_grouped.rename(columns={"date": "timestamp", "P&L": "value"}, inplace=True)
        combined_grouped["name"] = "Total P&L per day - All"
        combined_grouped["target_metric_id"] = target_metric_id
        
        combined_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    def profitable_trades_drawdown_per_direction(df, target_metric_id, metadata_name):
        """
        Calculate drawdown for trades that ended up profitable, separated by direction (Long/Short)
        """
        # Pivot the data to get P&L, Drawdown, and Direction as separate columns
        pivoted_df = df.pivot(index='timestamp', columns='name', values='value').reset_index()
        
        # Ensure columns are numeric
        pivoted_df['P&L'] = pd.to_numeric(pivoted_df['P&L'], errors='coerce')
        pivoted_df['Drawdown'] = pd.to_numeric(pivoted_df['Drawdown'], errors='coerce')
        pivoted_df["timestamp"] = pd.to_datetime(pivoted_df["timestamp"])
        pivoted_df["date"] = pivoted_df["timestamp"].dt.date
        
        # Filter for profitable trades only (P&L > 0)
        profitable_df = pivoted_df[pivoted_df['P&L'] > 0].copy()
        
        if profitable_df.empty:
            return
        
        # Process Long profitable trades
        long_profitable_df = profitable_df[profitable_df['Direction'] == 'Long'].copy()
        if not long_profitable_df.empty:
            long_grouped = long_profitable_df.groupby("date")["Drawdown"].mean().reset_index()
            long_grouped.rename(columns={"date": "timestamp", "Drawdown": "value"}, inplace=True)
            long_grouped["name"] = "Profitable Trades Drawdown - Long"
            long_grouped["target_metric_id"] = target_metric_id
            
            long_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
        
        # Process Short profitable trades
        short_profitable_df = profitable_df[profitable_df['Direction'] == 'Short'].copy()
        if not short_profitable_df.empty:
            short_grouped = short_profitable_df.groupby("date")["Drawdown"].mean().reset_index()
            short_grouped.rename(columns={"date": "timestamp", "Drawdown": "value"}, inplace=True)
            short_grouped["name"] = "Profitable Trades Drawdown - Short"
            short_grouped["target_metric_id"] = target_metric_id
            
            short_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
        
        # Process All profitable trades (combined)
        all_profitable_grouped = profitable_df.groupby("date")["Drawdown"].mean().reset_index()
        all_profitable_grouped.rename(columns={"date": "timestamp", "Drawdown": "value"}, inplace=True)
        all_profitable_grouped["name"] = "Profitable Trades Drawdown - All"
        all_profitable_grouped["target_metric_id"] = target_metric_id
        
        all_profitable_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    def hit_rate_per_direction(df, target_metric_id, metadata_name):
        """
        Calculate hit rate (percentage of profitable trades) separated by direction (Long/Short)
        """
        # Pivot the data to get P&L and Direction as separate columns
        pivoted_df = df.pivot(index='timestamp', columns='name', values='value').reset_index()
        
        # Ensure P&L is numeric
        pivoted_df['P&L'] = pd.to_numeric(pivoted_df['P&L'], errors='coerce')
        pivoted_df["timestamp"] = pd.to_datetime(pivoted_df["timestamp"])
        pivoted_df["date"] = pivoted_df["timestamp"].dt.date
        
        # Calculate hit rate (profitable trades percentage)
        pivoted_df["hit_rate"] = ((pivoted_df["P&L"] > 0).astype(int)) * 100
        
        # Process Long trades
        long_df = pivoted_df[pivoted_df['Direction'] == 'Long'].copy()
        if not long_df.empty:
            long_grouped = long_df.groupby("date")["hit_rate"].mean().reset_index()
            long_grouped.rename(columns={"date": "timestamp", "hit_rate": "value"}, inplace=True)
            long_grouped["name"] = "Hit Rate - Long"
            long_grouped["target_metric_id"] = target_metric_id
            
            long_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
        
        # Process Short trades
        short_df = pivoted_df[pivoted_df['Direction'] == 'Short'].copy()
        if not short_df.empty:
            short_grouped = short_df.groupby("date")["hit_rate"].mean().reset_index()
            short_grouped.rename(columns={"date": "timestamp", "hit_rate": "value"}, inplace=True)
            short_grouped["name"] = "Hit Rate - Short"
            short_grouped["target_metric_id"] = target_metric_id
            
            short_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    def pnl_split_wins_losses(df, target_metric_id, metadata_name):
        """
        Split P&L into three separate metrics: All P&L, Losses Only, and Wins Only
        """
        # Handle both single metadata (timestamp, value) and multiple metadata (timestamp, name, value) formats
        if 'name' in df.columns:
            # Multiple metadata format - need to pivot
            pivoted_df = df.pivot(index='timestamp', columns='name', values='value').reset_index()
            # Ensure P&L column exists after pivot
            if 'P&L' not in pivoted_df.columns:
                logger.warning(f"P&L column not found after pivot for target_metric_id {target_metric_id}")
                return
            pnl_data = pivoted_df['P&L']
        else:
            # Single metadata format - data is already in the right format
            pivoted_df = df.copy()
            pnl_data = df['value']
        
        # Ensure P&L is numeric
        pnl_data = pd.to_numeric(pnl_data, errors='coerce')
        pivoted_df["timestamp"] = pd.to_datetime(pivoted_df["timestamp"])
        pivoted_df["date"] = pivoted_df["timestamp"].dt.date
        
        # Add the P&L data to the dataframe if it's not already there
        if 'name' not in df.columns:
            pivoted_df['P&L'] = pnl_data
        
        # 2. Losses Only (negative P&L)
        losses_df = pivoted_df[pnl_data < 0].copy()
        if not losses_df.empty:
            losses_grouped = losses_df.groupby("date").agg({'P&L': 'mean'}).reset_index()
            losses_grouped.rename(columns={"date": "timestamp", "P&L": "value"}, inplace=True)
            losses_grouped["name"] = "P&L - Losses Only"
            losses_grouped["target_metric_id"] = target_metric_id
            losses_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
        
        # 3. Wins Only (positive P&L)
        wins_df = pivoted_df[pnl_data > 0].copy()
        if not wins_df.empty:
            wins_grouped = wins_df.groupby("date").agg({'P&L': 'mean'}).reset_index()
            wins_grouped.rename(columns={"date": "timestamp", "P&L": "value"}, inplace=True)
            wins_grouped["name"] = "P&L - Wins Only"
            wins_grouped["target_metric_id"] = target_metric_id
            wins_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    def average_duration_by_type(df, target_metric_id, metadata_name):
        """
        Calculate average duration in minutes for different trade types:
        - Long trades
        - Short trades
        - Winning trades
        - Losing trades
        """
        # Handle both single metadata and multiple metadata formats
        if 'name' in df.columns:
            # Multiple metadata format - need to pivot
            pivoted_df = df.pivot(index='timestamp', columns='name', values='value').reset_index()
            
            # Check required columns exist
            required_cols = ['P&L', 'Direction', 'Duration in Minutes']
            missing_cols = [col for col in required_cols if col not in pivoted_df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns {missing_cols} for average_duration_by_type target_metric_id {target_metric_id}")
                return
            
            # Ensure columns are numeric/correct type
            pivoted_df['P&L'] = pd.to_numeric(pivoted_df['P&L'], errors='coerce')
            pivoted_df['Duration in Minutes'] = pd.to_numeric(pivoted_df['Duration in Minutes'], errors='coerce')
            pivoted_df["timestamp"] = pd.to_datetime(pivoted_df["timestamp"])
            pivoted_df["date"] = pivoted_df["timestamp"].dt.date
            
            # Drop rows with invalid data
            pivoted_df = pivoted_df.dropna(subset=['P&L', 'Duration in Minutes', 'Direction'])
            
            if pivoted_df.empty:
                logger.warning(f"No valid data after cleaning for average_duration_by_type target_metric_id {target_metric_id}")
                return
            
            # Calculate average duration for Long trades
            long_df = pivoted_df[pivoted_df['Direction'] == 'Long'].copy()
            if not long_df.empty:
                long_grouped = long_df.groupby("date")["Duration in Minutes"].mean().reset_index()
                long_grouped.rename(columns={"date": "timestamp", "Duration in Minutes": "value"}, inplace=True)
                long_grouped["name"] = "Avg Duration - Long"
                long_grouped["target_metric_id"] = target_metric_id
                long_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
            
            # Calculate average duration for Short trades
            short_df = pivoted_df[pivoted_df['Direction'] == 'Short'].copy()
            if not short_df.empty:
                short_grouped = short_df.groupby("date")["Duration in Minutes"].mean().reset_index()
                short_grouped.rename(columns={"date": "timestamp", "Duration in Minutes": "value"}, inplace=True)
                short_grouped["name"] = "Avg Duration - Short"
                short_grouped["target_metric_id"] = target_metric_id
                short_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
            
            # Calculate average duration for Winning trades
            winning_df = pivoted_df[pivoted_df['P&L'] > 0].copy()
            if not winning_df.empty:
                win_grouped = winning_df.groupby("date")["Duration in Minutes"].mean().reset_index()
                win_grouped.rename(columns={"date": "timestamp", "Duration in Minutes": "value"}, inplace=True)
                win_grouped["name"] = "Avg Duration - Winners"
                win_grouped["target_metric_id"] = target_metric_id
                win_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
            
            # Calculate average duration for Losing trades
            losing_df = pivoted_df[pivoted_df['P&L'] <= 0].copy()
            if not losing_df.empty:
                loss_grouped = losing_df.groupby("date")["Duration in Minutes"].mean().reset_index()
                loss_grouped.rename(columns={"date": "timestamp", "Duration in Minutes": "value"}, inplace=True)
                loss_grouped["name"] = "Avg Duration - Losers"
                loss_grouped["target_metric_id"] = target_metric_id
                loss_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
        else:
            logger.warning(f"Single metadata format not supported for average_duration_by_type target_metric_id {target_metric_id}")
            return

    def drawdown_per_direction(df, target_metric_id, metadata_name):
        """
        Calculate average drawdown separated by direction (Long/Short)
        This includes all trades, not just profitable ones
        """
        # Pivot the data to get P&L, Drawdown, and Direction as separate columns
        pivoted_df = df.pivot(index='timestamp', columns='name', values='value').reset_index()
        
        # Check required columns exist
        required_cols = ['P&L', 'Drawdown', 'Direction']
        missing_cols = [col for col in required_cols if col not in pivoted_df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns {missing_cols} for drawdown_per_direction target_metric_id {target_metric_id}")
            return
        
        # Ensure columns are numeric
        pivoted_df['P&L'] = pd.to_numeric(pivoted_df['P&L'], errors='coerce')
        pivoted_df['Drawdown'] = pd.to_numeric(pivoted_df['Drawdown'], errors='coerce')
        pivoted_df["timestamp"] = pd.to_datetime(pivoted_df["timestamp"])
        pivoted_df["date"] = pivoted_df["timestamp"].dt.date
        
        # Drop rows with invalid data
        pivoted_df = pivoted_df.dropna(subset=['P&L', 'Drawdown', 'Direction'])
        
        if pivoted_df.empty:
            logger.warning(f"No valid data after cleaning for drawdown_per_direction target_metric_id {target_metric_id}")
            return
        
        # Process Long trades drawdown
        long_df = pivoted_df[pivoted_df['Direction'] == 'Long'].copy()
        if not long_df.empty:
            long_grouped = long_df.groupby("date")["Drawdown"].mean().reset_index()
            long_grouped.rename(columns={"date": "timestamp", "Drawdown": "value"}, inplace=True)
            long_grouped["name"] = "Drawdown - Long"
            long_grouped["target_metric_id"] = target_metric_id
            
            long_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
        
        # Process Short trades drawdown
        short_df = pivoted_df[pivoted_df['Direction'] == 'Short'].copy()
        if not short_df.empty:
            short_grouped = short_df.groupby("date")["Drawdown"].mean().reset_index()
            short_grouped.rename(columns={"date": "timestamp", "Drawdown": "value"}, inplace=True)
            short_grouped["name"] = "Drawdown - Short"
            short_grouped["target_metric_id"] = target_metric_id
            
            short_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    def pnl_per_trade_per_direction_per_hour(df, target_metric_id, metadata_name):
        """Calculate P&L per trade per direction per trading hour (when trade was entered)"""
        # Pivot the dataframe to have metadata as columns
        pivoted_df = df.pivot(index='timestamp', columns='name', values='value').reset_index()
        
        # Validate required columns exist
        required_cols = ['P&L', 'Direction']
        missing_cols = [col for col in required_cols if col not in pivoted_df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns {missing_cols} for pnl_per_trade_per_direction_per_hour target_metric_id {target_metric_id}")
            return
        
        # Ensure columns are numeric and clean data
        pivoted_df['P&L'] = pd.to_numeric(pivoted_df['P&L'], errors='coerce')
        pivoted_df["timestamp"] = pd.to_datetime(pivoted_df["timestamp"])
        
        # Extract trading hour from timestamp (when trade was entered)
        pivoted_df["trading_hour"] = pivoted_df["timestamp"].dt.hour
        
        # Drop rows with invalid data
        pivoted_df = pivoted_df.dropna(subset=['P&L', 'Direction'])
        
        if pivoted_df.empty:
            logger.warning(f"No valid data after cleaning for pnl_per_trade_per_direction_per_hour target_metric_id {target_metric_id}")
            return
        
        # Process Long trades P&L per trading hour
        long_df = pivoted_df[pivoted_df['Direction'] == 'Long'].copy()
        if not long_df.empty:
            # Group by trading hour and calculate mean P&L per trade
            long_grouped = long_df.groupby("trading_hour")["P&L"].mean().reset_index()
            long_grouped.rename(columns={"trading_hour": "timestamp", "P&L": "value"}, inplace=True)
            
            # Convert hour back to a timestamp format for consistency (using a base date)
            long_grouped["timestamp"] = pd.to_datetime('2000-01-01') + pd.to_timedelta(long_grouped["timestamp"], unit='h')
            
            long_grouped["name"] = "P&L per trade per hour - Long"
            long_grouped["target_metric_id"] = target_metric_id
            
            long_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)
        
        # Process Short trades P&L per trading hour  
        short_df = pivoted_df[pivoted_df['Direction'] == 'Short'].copy()
        if not short_df.empty:
            # Group by trading hour and calculate mean P&L per trade
            short_grouped = short_df.groupby("trading_hour")["P&L"].mean().reset_index()
            short_grouped.rename(columns={"trading_hour": "timestamp", "P&L": "value"}, inplace=True)
            
            # Convert hour back to a timestamp format for consistency (using a base date)
            short_grouped["timestamp"] = pd.to_datetime('2000-01-01') + pd.to_timedelta(short_grouped["timestamp"], unit='h')
            
            short_grouped["name"] = "P&L per trade per hour - Short"
            short_grouped["target_metric_id"] = target_metric_id
            
            short_grouped.to_sql("engineered_target_metric_row", con=db_session.connection(), if_exists="append", index=False)

    backtest_data_functions = {
        "hit_rate" : hit_rate,
        "raw" : raw,
        "pnl_per_day" : pnl_per_day,
        "P&L_per_direction" : pnl_per_direction,
        "total_pnl_per_day_per_direction" : total_pnl_per_day_per_direction,
        "profitable_trades_drawdown_per_direction" : profitable_trades_drawdown_per_direction,
        "hit_rate_per_direction" : hit_rate_per_direction,
        "pnl_split_wins_losses" : pnl_split_wins_losses,
        "average_duration_by_type" : average_duration_by_type,
        "drawdown_per_direction" : drawdown_per_direction,
        "pnl_per_trade_per_direction_per_hour" : pnl_per_trade_per_direction_per_hour
    }

    if len(metadata_names) == 1:
        metadata_name = metadata_names[0]

        query = text(f"""
            SELECT 
                tmr.timestamp,
                mdr.value
            FROM target_metric_metadata_row mdr
            JOIN target_metric_row tmr ON mdr.target_metric_row_id = tmr.id
            WHERE tmr.target_metric_id = :target_metric_id
            AND mdr.name = :metadata_name;
        """)

        result = db_session.execute(query, {
            "target_metric_id": target_metric_id,
            "metadata_name": metadata_name
        })
    else:
        placeholders = ", ".join([f":name_{i}" for i in range(len(metadata_names))])

        query = text(f"""
            SELECT 
                tmr.timestamp,
                mdr.name,
                mdr.value
            FROM target_metric_metadata_row mdr
            JOIN target_metric_row tmr ON mdr.target_metric_row_id = tmr.id
            WHERE tmr.target_metric_id = :target_metric_id
            AND mdr.name IN ({placeholders});
        """)

        params = {"target_metric_id": target_metric_id}
        params.update({f"name_{i}": name for i, name in enumerate(metadata_names)})

        result = db_session.execute(query, params)

    rows = result.fetchall()
    columns = result.keys()

    df = pd.DataFrame(rows, columns=columns)
    
    # Validate that we have data to process
    if df.empty:
        logger.warning(f"No data found for target_metric_id {target_metric_id} with metadata {metadata_names}")
        return
    
    # Validate that the function exists
    if function not in backtest_data_functions:
        logger.error(f"Unknown engineering function: {function}")
        raise ValueError(f"Engineering function '{function}' not supported")
    
    # Validate required columns exist
    if len(metadata_names) == 1:
        if 'value' not in df.columns or 'timestamp' not in df.columns:
            logger.warning(f"Missing required columns for target_metric_id {target_metric_id}")
            return
    else:
        if 'name' not in df.columns or 'value' not in df.columns or 'timestamp' not in df.columns:
            logger.warning(f"Missing required columns for multi-metadata processing of target_metric_id {target_metric_id}")
            return
    
    try:
        # Execute the engineering function
        backtest_data_functions[function](df, target_metric_id, metadata_names[0])
        logger.info(f"Successfully processed target_metric_id {target_metric_id} with function '{function}'")
    except Exception as e:
        logger.error(f"Error processing target_metric_id {target_metric_id} with function '{function}': {str(e)}")
        raise

def handle_tradingview_backtest_data(db, filename, target_metric_id):
    def get_target_metric_row_df(db, df):
        entry_mask = df["Type"].str.contains("Entry", case=False)
        entries = df[entry_mask].copy().reset_index(drop=True)

        # Create DataFrame for bulk insert
        timestamp_df = pd.DataFrame({
            "timestamp": entries["Date/Time"],
            "target_metric_id": target_metric_id
        })
        
        # Use the session's connection for bulk insert while maintaining transaction
        timestamp_df.to_sql("target_metric_row", con=db_session.connection(), if_exists="append", index=False)

        result = db.execute(text("SELECT id, timestamp FROM target_metric_row WHERE target_metric_id = :target_metric_id"), 
                           {"target_metric_id": target_metric_id})

        rows = result.fetchall()
        columns = result.keys()

        return pd.DataFrame(rows, columns=columns).rename(columns={"id": "target_metric_row_id"})

    def get_pnl_metadata(df, target_metric_id, target_metric_row_df, db_session):
        entry_mask = df["Type"].str.contains("Entry", case=False)
        exit_mask = df["Type"].str.contains("Exit", case=False)

        entries = df[entry_mask].copy().reset_index(drop=True)
        exits = df[exit_mask].copy().reset_index(drop=True)

        try:
            df =  pd.DataFrame({
                "timestamp": entries["Date/Time"],
                "value": exits["P&L USD"],
                "name" : "P&L"
            })
        except:
            df =  pd.DataFrame({
                "timestamp": entries["Date/Time"],
                "value": exits["P&L EUR"],
                "name" : "P&L"
            })

        merged_df = pd.merge(df, target_metric_row_df, on="timestamp", how="inner")

        merged_df = merged_df.drop(columns=["timestamp"])

        merged_df.to_sql("target_metric_metadata_row", con=db_session.connection(), if_exists="append", index=False)

    def get_run_up_data(df, target_metric_id, target_metric_row_df, db_session):
        entry_mask = df["Type"].str.contains("Entry", case=False)
        exit_mask = df["Type"].str.contains("Exit", case=False)

        entries = df[entry_mask].copy().reset_index(drop=True)
        exits = df[exit_mask].copy().reset_index(drop=True)

        try:
            df =  pd.DataFrame({
                "timestamp": entries["Date/Time"],
                "value": exits["Run-up USD"],
                "name" : "Run-up"
            })
        except:
            df =  pd.DataFrame({
                "timestamp": entries["Date/Time"],
                "value": exits["Run-up EUR"],
                "name" : "Run-up"
            })

        merged_df = pd.merge(df, target_metric_row_df, on="timestamp", how="inner")

        merged_df = merged_df.drop(columns=["timestamp"])

        merged_df.to_sql("target_metric_metadata_row", con=db_session.connection(), if_exists="append", index=False)

    def get_drawdown_data(df, target_metric_id, target_metric_row_df, db_session):
        entry_mask = df["Type"].str.contains("Entry", case=False)
        exit_mask = df["Type"].str.contains("Exit", case=False)

        entries = df[entry_mask].copy().reset_index(drop=True)
        exits = df[exit_mask].copy().reset_index(drop=True)

        try:
            df =  pd.DataFrame({
                "timestamp": entries["Date/Time"],
                "value": exits["Drawdown USD"],
                "name" : "Drawdown"
            })
        except:
            df =  pd.DataFrame({
                "timestamp": entries["Date/Time"],
                "value": exits["Drawdown EUR"],
                "name" : "Drawdown"
            })

        merged_df = pd.merge(df, target_metric_row_df, on="timestamp", how="inner")

        merged_df = merged_df.drop(columns=["timestamp"])

        merged_df.to_sql("target_metric_metadata_row", con=db_session.connection(), if_exists="append", index=False)

    def get_direction_data(df, target_metric_id, target_metric_row_df, db_session):
        entry_mask = df["Type"].str.contains("Entry", case=False)
        exit_mask = df["Type"].str.contains("Exit", case=False)

        entries = df[entry_mask].copy().reset_index(drop=True)
        exits = df[exit_mask].copy().reset_index(drop=True)

        df =  pd.DataFrame({
            "timestamp": entries["Date/Time"],
            "value": entries["Signal"],
            "name" : "Direction"
        })

        merged_df = pd.merge(df, target_metric_row_df, on="timestamp", how="inner")

        merged_df = merged_df.drop(columns=["timestamp"])

        merged_df.to_sql("target_metric_metadata_row", con=db_session.connection(), if_exists="append", index=False)

    def get_timeframe_data(df, timeframe_df, target_metric_id, target_metric_row_df, db_session):
        timeframe = timeframe_df.loc[timeframe_df["name"] == "Timeframe", "value"].values[0]

        entry_mask = df["Type"].str.contains("Entry", case=False)
        exit_mask = df["Type"].str.contains("Exit", case=False)

        entries = df[entry_mask].copy().reset_index(drop=True)
        exits = df[exit_mask].copy().reset_index(drop=True)

        df =  pd.DataFrame({
            "timestamp": entries["Date/Time"],
            "value": timeframe,
            "name" : "Timeframe"
        })

        merged_df = pd.merge(df, target_metric_row_df, on="timestamp", how="inner")

        merged_df = merged_df.drop(columns=["timestamp"])

        merged_df.to_sql("target_metric_metadata_row", con=db_session.connection(), if_exists="append", index=False)

    def get_duration_data(df, target_metric_id, target_metric_row_df, db_session):
        df["Date/Time"] = pd.to_datetime(df["Date/Time"])

        entry_mask = df["Type"].str.contains("Entry", case=False)
        exit_mask = df["Type"].str.contains("Exit", case=False)

        entries = df[entry_mask].copy().reset_index(drop=True)
        exits = df[exit_mask].copy().reset_index(drop=True)

        durations = (exits["Date/Time"] - entries["Date/Time"]).dt.total_seconds() / 60

        result_df = pd.DataFrame({
            "timestamp": entries["Date/Time"],
            "value": durations,
            "name" : "Duration in Minutes"
        })

        merged_df = pd.merge(result_df, target_metric_row_df, on="timestamp", how="inner")

        merged_df = merged_df.drop(columns=["timestamp"])

        merged_df.to_sql("target_metric_metadata_row", con=db_session.connection(), if_exists="append", index=False)

    df = pd.read_excel(filename, sheet_name="List of trades").sort_values(by=["Trade #", "Type"], ascending=[True, True])
    timeframe_df = pd.read_excel(filename, sheet_name="Properties")

    target_metric_row_df = get_target_metric_row_df(db, df)

    get_pnl_metadata(df, target_metric_id, target_metric_row_df, db)
    get_run_up_data(df, target_metric_id, target_metric_row_df, db)
    get_drawdown_data(df, target_metric_id, target_metric_row_df, db)
    get_direction_data(df, target_metric_id, target_metric_row_df, db)
    get_timeframe_data(df, timeframe_df, target_metric_id, target_metric_row_df, db)
    get_duration_data(df, target_metric_id, target_metric_row_df, db)

    run_backtest_function(db, 'raw', target_metric_id, ['P&L'])
    run_backtest_function(db, 'raw', target_metric_id, ['Run-up'])
    run_backtest_function(db, 'raw', target_metric_id, ['Drawdown'])
    run_backtest_function(db, 'raw', target_metric_id, ['Duration in Minutes'])

    run_backtest_function(db, 'hit_rate', target_metric_id, ['P&L'])
    run_backtest_function(db, 'pnl_per_day', target_metric_id, ['P&L'])
    run_backtest_function(db, 'P&L_per_direction', target_metric_id, ['P&L', 'Direction'])
    run_backtest_function(db, 'total_pnl_per_day_per_direction', target_metric_id, ['P&L', 'Direction'])
    run_backtest_function(db, 'profitable_trades_drawdown_per_direction', target_metric_id, ['P&L', 'Drawdown', 'Direction'])
    run_backtest_function(db, 'drawdown_per_direction', target_metric_id, ['P&L', 'Drawdown', 'Direction'])
    run_backtest_function(db, 'hit_rate_per_direction', target_metric_id, ['P&L', 'Direction'])
    run_backtest_function(db, 'pnl_split_wins_losses', target_metric_id, ['P&L'])
    run_backtest_function(db, 'average_duration_by_type', target_metric_id, ['P&L', 'Direction', 'Duration in Minutes'])


def insert_new_target_metric_with_session(filename, source, function, target_metric_name, sub_category):
    """New version using context manager for better connection handling"""
    with get_session() as db:
        # Use parameterized queries to avoid SQL injection
        if function == "Backtest":
            result = db.execute(text("SELECT id FROM target_metric_category WHERE name = :name"), {"name": "backtested_setups"})
            category_row = result.fetchone()
            if not category_row:
                raise ValueError("Category 'backtested_setups' not found")
            target_metric_category_id = category_row[0]
        elif function == "12 month forward return":
            result = db.execute(text("SELECT id FROM target_metric_category WHERE name = :name"), {"name": "Long term portfolio allocation"})
            category_row = result.fetchone()
            if not category_row:
                raise ValueError("Category 'Long term portfolio allocation' not found")
            target_metric_category_id = category_row[0]
        else:
            raise ValueError(f"Unknown function: {function}")
        
        # Insert target metric
        columns = "name, target_metric_category_id, sub_category"
        placeholders = ":name, :target_metric_category_id, :sub_category"
        query = text(f"INSERT INTO target_metric ({columns}) VALUES ({placeholders}) RETURNING ID;")
        
        target_metric_data = {
            "name": target_metric_name,
            "target_metric_category_id": target_metric_category_id,
            "sub_category": sub_category
        }
        
        result = db.execute(query, target_metric_data)
        target_metric_id = result.fetchone()[0]
        
        if not target_metric_id:
            raise ValueError("Failed to create target metric")

        if source == "TradingView":
            if function == "Backtest":
                df = handle_tradingview_backtest_data(db, filename, target_metric_id)
            elif function == "12 month forward return":
                df = calculate_12_month_return(filename, target_metric_id, db)
        
        return target_metric_id

def insert_new_target_metric(db, filename, source, function, target_metric_name, sub_category):
    """Legacy function - redirects to new context manager version"""
    return insert_new_target_metric_with_session(filename, source, function, target_metric_name, sub_category)

def get_target_metric_data(metric_id, engineered_target_metric_name):
    db = get_db()

    query = text(f"SELECT * FROM engineered_target_metric_row WHERE target_metric_id = {metric_id} AND name = '{engineered_target_metric_name}';")
    
    result = db.execute(query)

    rows = result.fetchall()
    columns = result.keys()

    df = pd.DataFrame(rows, columns=columns)

    df['timestamp'] = df['timestamp'].dt.date
    daily_avg = df.groupby('timestamp')['value'].mean().reset_index()

    close_db(db)

    return daily_avg

def get_regime_data(regime_model_id):
    db = get_db()

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
    """)

    result = db.execute(query, {"regime_model_id": regime_model_id})
    df = pd.DataFrame(result.fetchall(), columns=["timestamp", "regime_name", "probability"])

    df_max = df.sort_values("probability", ascending=False).drop_duplicates("timestamp")

    regime_counts = df_max["regime_name"].value_counts().to_dict()

    df = df_max[["timestamp", "regime_name"]].sort_values("timestamp").reset_index(drop=True)

    close_db(db)

    return df, regime_counts

def get_historical_regime_data(regime_model_id):
    """Get historical regime probability data for time series visualization"""
    db = get_db()

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
        close_db(db)
        return pd.DataFrame(), {}

    # Get regime counts based on top regime per timestamp
    df_max = df.sort_values("probability", ascending=False).drop_duplicates("timestamp")
    regime_counts = df_max["regime_name"].value_counts().to_dict()

    # For historical analysis, we want all probability data, not just the top regime
    # Pivot to create time series with regime columns
    df_pivot = df.pivot(index='timestamp', columns='regime_name', values='probability')
    df_pivot = df_pivot.fillna(0)  # Fill missing probabilities with 0
    df_pivot = df_pivot.reset_index()

    close_db(db)

    return df_pivot, regime_counts

def get_mean_metric_per_regime(target_metric_id, model_id, engineered_target_metric_name):
    db = get_db()

    target_metric_df = get_target_metric_data(target_metric_id, engineered_target_metric_name)
    regime_df, regime_counts = get_regime_data(model_id)

    target_metric_df["timestamp"] = pd.to_datetime(target_metric_df["timestamp"])
    regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"])

    regime_df = regime_df.sort_values("timestamp").reset_index(drop=True)
    regime_df["regime_name_shifted"] = regime_df["regime_name"].shift(1)

    regime_df = regime_df.dropna(subset=["regime_name_shifted"])
    regime_df = regime_df[["timestamp", "regime_name_shifted"]].rename(columns={"regime_name_shifted": "regime_name"})

    df_merged = pd.merge(target_metric_df, regime_df, on="timestamp", how="left")

    df_merged = df_merged.dropna(subset=["regime_name"])

    mean_per_regime = df_merged.groupby("regime_name")["value"].mean().reset_index()

    close_db(db)

    return mean_per_regime

def get_current_regime(model_id):
    db = get_db()

    query = text("""
        SELECT DISTINCT ON (rr.regime_id)
            rr.regime_id,
            r.name AS regime_name,
            rr.timestamp,
            rr.probability
        FROM regime_row rr
        JOIN regime r ON rr.regime_id = r.id
        WHERE r.regime_model_id = :model_id
        ORDER BY rr.regime_id, rr.timestamp DESC
    """)

    result = db.execute(query, {"model_id": model_id})
    df = pd.DataFrame(result.fetchall(), columns=["regime_id", "regime_name", "timestamp", "probability"])

    close_db(db)

    return df

def get_batch_setup_list_data(target_metric_ids, model_id, engineered_target_metric_names):
    """
    Ultra-optimized single SQL query for setup list data
    """
    db = get_db()
    
    target_ids_str = ",".join(map(str, target_metric_ids))
    metric_names_str = "','".join(engineered_target_metric_names)
    
    # Single massive query that does everything at the database level
    ultra_query = db.execute(text(f"""
        WITH regime_probabilities AS (
            SELECT 
                rr.timestamp,
                r.name as regime_name,
                rr.probability,
                ROW_NUMBER() OVER (PARTITION BY rr.timestamp ORDER BY rr.probability DESC) as rank
            FROM regime_row rr
            JOIN regime r ON rr.regime_id = r.id
            WHERE r.regime_model_id = {model_id}
        ),
        top_regimes AS (
            SELECT timestamp, regime_name, probability
            FROM regime_probabilities 
            WHERE rank = 1
        ),
        shifted_regimes AS (
            SELECT 
                timestamp,
                LAG(regime_name) OVER (ORDER BY timestamp) as regime_name_shifted,
                LAG(probability) OVER (ORDER BY timestamp) as probability_shifted
            FROM top_regimes
        ),
        current_regimes AS (
            SELECT DISTINCT ON (regime_name)
                regime_name,
                probability
            FROM regime_probabilities
            WHERE rank = 1
            ORDER BY regime_name, timestamp DESC
        ),
        metric_regime_means AS (
            SELECT 
                etmr.target_metric_id,
                etmr.name as engineered_metric_name,
                sr.regime_name_shifted as regime_name,
                AVG(etmr.value) as mean_value
            FROM engineered_target_metric_row etmr
            JOIN shifted_regimes sr ON DATE(etmr.timestamp) = DATE(sr.timestamp)
            WHERE etmr.target_metric_id IN ({target_ids_str})
            AND etmr.name IN ('{metric_names_str}')
            AND sr.regime_name_shifted IS NOT NULL
            GROUP BY etmr.target_metric_id, etmr.name, sr.regime_name_shifted
        ),
        expected_values AS (
            SELECT 
                mrm.target_metric_id,
                mrm.engineered_metric_name,
                SUM(mrm.mean_value * cr.probability) as expected_value
            FROM metric_regime_means mrm
            JOIN current_regimes cr ON mrm.regime_name = cr.regime_name
            GROUP BY mrm.target_metric_id, mrm.engineered_metric_name
        )
        SELECT 
            tm.name,
            tm.sub_category,
            mdr.value as timeframe,
            ev.engineered_metric_name,
            ev.expected_value
        FROM expected_values ev
        JOIN target_metric tm ON ev.target_metric_id = tm.id
        JOIN target_metric_row tmr ON tm.id = tmr.target_metric_id
        JOIN target_metric_metadata_row mdr ON mdr.target_metric_row_id = tmr.id
        WHERE mdr.name = 'Timeframe'
        ORDER BY tm.name, tm.sub_category, mdr.value, ev.engineered_metric_name
    """)).fetchall()
    
    # Build result dictionary from single query
    result = {}
    for name, sub_category, timeframe, metric_name, expected_val in ultra_query:
        key = (name, sub_category, timeframe)
        if key not in result:
            result[key] = {}
        result[key][metric_name] = expected_val
    
    close_db(db)
    return result

def get_live_regime_data(target_metric_id, model_id, engineered_target_metric_name):
    mean_per_regime = get_mean_metric_per_regime(target_metric_id, model_id, engineered_target_metric_name)
    current_regime = get_current_regime(model_id)

    df = pd.merge(mean_per_regime, current_regime, on="regime_name", how="inner").drop(columns=["timestamp", "regime_id"])

    return df