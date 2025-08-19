import pandas as pd
import numpy as np

def standardize_column(df):
    possible_columns = ["value", "close", "open", "high", "low", "volume"]
    for col in possible_columns:
        if col in df.columns:
            df = df.rename(columns={col: "value"})
            return df

def rsi(df):
    data = df["close"]

    period = 14
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.dropna()

def rsi_15_minute(df, period=14):
    """
    Calculate RSI on 15-minute resampled close prices.
    Expects df with columns: 'timestamp', 'close'
    Returns Series with RSI values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    original_timestamps = df['timestamp'].copy()
    df = df.set_index('timestamp')

    # Resample to 15-minute closes
    close_15m = df['close'].resample('15min').last()

    # RSI calculation
    delta = close_15m.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Forward fill the RSI values to propagate them to all timestamps
    rsi = rsi.ffill()
    
    # Reindex to original timestamps to maintain alignment
    rsi_aligned = rsi.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(rsi_aligned.values)

def rsi_1_hour(df, period=14):
    """
    Calculate RSI on 1-hour resampled close prices.
    Expects df with columns: 'timestamp', 'close'
    Returns Series with RSI values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    original_timestamps = df['timestamp'].copy()
    df = df.set_index('timestamp')

    # Resample to 1-hour closes
    close_1h = df['close'].resample('1h').last()

    # RSI calculation
    delta = close_1h.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Forward fill the RSI values to propagate them to all timestamps
    rsi = rsi.ffill()
    
    # Reindex to original timestamps to maintain alignment
    rsi_aligned = rsi.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(rsi_aligned.values)

def yoy(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    if "close" in df.columns:
        df.rename(columns={"close": "value"}, inplace=True)

    frequency = pd.infer_freq(df.index)

    if str(frequency).startswith("Q"):
        df["yoy"] = df["value"].pct_change(periods=4) * 100
    elif str(frequency).startswith("M"):
        df["yoy"] = df["value"].pct_change(periods=12) * 100
    elif str(frequency).startswith("D"):
        df["yoy"] = df["value"].pct_change(periods=365) * 100
    else:
        raise ValueError("Frequency not supported or undetectable.")

    df = df[["yoy"]].dropna().reset_index()

    return df["yoy"]

def raw(df):
    return df[[col for col in df.columns if col != "timestamp"]]

def return_30d(df):
    series = df["close"]

    return series.pct_change(periods=30) * 100

def return_1d(df):
    series = df["close"]

    return series.pct_change(periods=1) * 100

def correlation_2_assets(df):
    return df['close'].rolling(window=30).corr(df['close_1']).dropna()

def return_ytd(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    df["year"] = df["timestamp"].dt.year

    first_values = df.groupby("year").first()["close"]

    df["start_of_year_value"] = df["year"].map(first_values)

    df["ytd_return"] = (df["close"] / df["start_of_year_value"] - 1) * 100

    return df["ytd_return"]

def distance_from_sma_200(df):
    df = df.copy()
    df = df.sort_values("timestamp")

    df["sma_200"] = df["close"].rolling(window=200).mean()
    df["distance_from_sma_200_%"] = ((df["close"] - df["sma_200"]) / df["sma_200"]) * 100
    df = df.dropna(subset=["sma_200"])

    return df["distance_from_sma_200_%"]

def calculate_atr_14(df):
    df = df.copy()
    df = df.sort_values("timestamp")

    df["previous_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = (df["high"] - df["previous_close"]).abs()
    df["tr3"] = (df["low"] - df["previous_close"]).abs()

    df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

    df["atr_14"] = df["true_range"].rolling(window=14).mean()

    return df["atr_14"]

def atr_15_minute(df, period=14):
    """
    Calculate ATR on 15-minute resampled OHLC data.
    Expects df with columns: 'timestamp', 'open', 'high', 'low', 'close'
    Returns Series with ATR values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute OHLC
    ohlc_15m = df[['open', 'high', 'low', 'close']].resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate True Range components
    ohlc_15m['previous_close'] = ohlc_15m['close'].shift(1)
    ohlc_15m['tr1'] = ohlc_15m['high'] - ohlc_15m['low']
    ohlc_15m['tr2'] = (ohlc_15m['high'] - ohlc_15m['previous_close']).abs()
    ohlc_15m['tr3'] = (ohlc_15m['low'] - ohlc_15m['previous_close']).abs()
    
    # Calculate True Range as max of the three components
    ohlc_15m['true_range'] = ohlc_15m[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR as rolling mean of True Range
    atr = ohlc_15m['true_range'].rolling(window=period).mean()
    
    # Forward fill the ATR values to propagate them to all timestamps
    atr = atr.ffill()
    
    # Reindex to original timestamps to maintain alignment
    atr_aligned = atr.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(atr_aligned.values)

def sma_5_slope_15_minute(df):
    """
    Calculate the slope of 5-period SMA on 15-minute resampled close prices.
    Expects df with columns: 'timestamp', 'close'
    Returns Series with slope values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute closes
    close_15m = df['close'].resample('15min').last()
    
    # Calculate 5-period SMA
    sma_5 = close_15m.rolling(window=5).mean()
    
    # Calculate slope (rate of change) of SMA
    # Slope = change in SMA / change in time (1 period = 1)
    slope = sma_5.diff()
    
    # Forward fill the slope values to propagate them to all timestamps
    slope = slope.ffill()
    
    # Reindex to original timestamps to maintain alignment
    slope_aligned = slope.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(slope_aligned.values)

def distance_from_sma_5_15_minute(df):
    """
    Calculate distance from 5-period SMA on 15-minute resampled close prices.
    Expects df with columns: 'timestamp', 'close'
    Returns Series with percentage distance values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute closes
    close_15m = df['close'].resample('15min').last()
    
    # Calculate 5-period SMA
    sma_5 = close_15m.rolling(window=5).mean()
    
    # Calculate percentage distance from SMA
    distance_pct = ((close_15m - sma_5) / sma_5) * 100
    
    # Forward fill the distance values to propagate them to all timestamps
    distance_pct = distance_pct.ffill()
    
    # Reindex to original timestamps to maintain alignment
    distance_aligned = distance_pct.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(distance_aligned.values)

def distance_from_sma_5_1_hour(df):
    """
    Calculate distance from 5-period SMA on 1-hour resampled close prices.
    Expects df with columns: 'timestamp', 'close'
    Returns Series with percentage distance values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour closes
    close_1h = df['close'].resample('1h').last()
    
    # Calculate 5-period SMA
    sma_5 = close_1h.rolling(window=5).mean()
    
    # Calculate percentage distance from SMA
    distance_pct = ((close_1h - sma_5) / sma_5) * 100
    
    # Forward fill the distance values to propagate them to all timestamps
    distance_pct = distance_pct.ffill()
    
    # Reindex to original timestamps to maintain alignment
    distance_aligned = distance_pct.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(distance_aligned.values)

def distance_from_sma_20_15_minute(df):
    """
    Calculate distance from 20-period SMA on 15-minute resampled close prices.
    Expects df with columns: 'timestamp', 'close'
    Returns Series with percentage distance values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute closes
    close_15m = df['close'].resample('15min').last()
    
    # Calculate 20-period SMA
    sma_20 = close_15m.rolling(window=20).mean()
    
    # Calculate percentage distance from SMA
    distance_pct = ((close_15m - sma_20) / sma_20) * 100
    
    # Forward fill the distance values to propagate them to all timestamps
    distance_pct = distance_pct.ffill()
    
    # Reindex to original timestamps to maintain alignment
    distance_aligned = distance_pct.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(distance_aligned.values)

def bollinger_band_width_15_minute(df, period=20, num_std=2):
    """
    Calculate Bollinger Band width on 15-minute resampled close prices.
    Width = (Upper Band - Lower Band) / Middle Band * 100
    Expects df with columns: 'timestamp', 'close'
    Returns Series with BB width values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute closes
    close_15m = df['close'].resample('15min').last()
    
    # Calculate Bollinger Bands components
    sma = close_15m.rolling(window=period).mean()  # Middle Band
    std = close_15m.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    # Calculate Bollinger Band width as percentage
    bb_width = ((upper_band - lower_band) / sma) * 100
    
    # Forward fill the width values to propagate them to all timestamps
    bb_width = bb_width.ffill()
    
    # Reindex to original timestamps to maintain alignment
    bb_width_aligned = bb_width.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(bb_width_aligned.values)

def rate_of_change_15_minute(df, period=10):
    """
    Calculate Rate of Change (ROC) on 15-minute resampled close prices.
    ROC = ((Current Price - Price N periods ago) / Price N periods ago) * 100
    Expects df with columns: 'timestamp', 'close'
    Returns Series with ROC values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute closes
    close_15m = df['close'].resample('15min').last()
    
    # Calculate Rate of Change
    roc = ((close_15m - close_15m.shift(period)) / close_15m.shift(period)) * 100
    
    # Forward fill the ROC values to propagate them to all timestamps
    roc = roc.ffill()
    
    # Reindex to original timestamps to maintain alignment
    roc_aligned = roc.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(roc_aligned.values)

def ichimoku_base_line_15_minute(df, period=26):
    """
    Calculate Ichimoku Base Line (Kijun-sen) on 15-minute resampled data.
    Base Line = (Highest High + Lowest Low) / 2 over period
    Expects df with columns: 'timestamp', 'high', 'low'
    Returns Series with Base Line values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute OHLC
    ohlc_15m = df[['open', 'high', 'low', 'close']].resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate Ichimoku Base Line (Kijun-sen)
    highest_high = ohlc_15m['high'].rolling(window=period).max()
    lowest_low = ohlc_15m['low'].rolling(window=period).min()
    base_line = (highest_high + lowest_low) / 2
    
    # Forward fill the Base Line values to propagate them to all timestamps
    base_line = base_line.ffill()
    
    # Reindex to original timestamps to maintain alignment
    base_line_aligned = base_line.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(base_line_aligned.values)

def macd_histogram_15_minute(df):
    """
    Calculate MACD histogram level on 15-minute resampled close prices.
    MACD histogram = MACD line - Signal line
    Expects df with columns: 'timestamp', 'close'
    Returns Series with MACD histogram values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute closes
    close_15m = df['close'].resample('15min').last()
    
    # Calculate MACD components
    # MACD line = 12-period EMA - 26-period EMA
    ema_12 = close_15m.ewm(span=12).mean()
    ema_26 = close_15m.ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    
    # Signal line = 9-period EMA of MACD line
    signal_line = macd_line.ewm(span=9).mean()
    
    # MACD histogram = MACD line - Signal line
    macd_histogram = macd_line - signal_line
    
    # Forward fill the histogram values
    macd_histogram = macd_histogram.ffill()
    
    # Reindex to original timestamps to maintain alignment
    histogram_aligned = macd_histogram.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(histogram_aligned.values)

def macd_histogram_1_hour(df):
    """
    Calculate MACD histogram level on 1-hour resampled close prices.
    MACD histogram = MACD line - Signal line
    Expects df with columns: 'timestamp', 'close'
    Returns Series with MACD histogram values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour closes
    close_1h = df['close'].resample('1h').last()
    
    # Calculate MACD components
    # MACD line = 12-period EMA - 26-period EMA
    ema_12 = close_1h.ewm(span=12).mean()
    ema_26 = close_1h.ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    
    # Signal line = 9-period EMA of MACD line
    signal_line = macd_line.ewm(span=9).mean()
    
    # MACD histogram = MACD line - Signal line
    macd_histogram = macd_line - signal_line
    
    # Forward fill the histogram values
    macd_histogram = macd_histogram.ffill()
    
    # Reindex to original timestamps to maintain alignment
    histogram_aligned = macd_histogram.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(histogram_aligned.values)

def historical_volatility_10_15_minute(df):
    """
    Calculate 10-period historical volatility on 15-minute resampled close prices.
    Historical volatility = standard deviation of log returns over 10 periods
    Expects df with columns: 'timestamp', 'close'
    Returns Series with volatility values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute closes
    close_15m = df['close'].resample('15min').last()
    
    # Calculate log returns
    log_returns = np.log(close_15m / close_15m.shift(1))
    
    # Calculate 10-period rolling standard deviation (historical volatility)
    historical_vol = log_returns.rolling(window=10).std()
    
    # Forward fill the volatility values
    historical_vol = historical_vol.ffill()
    
    # Reindex to original timestamps to maintain alignment
    vol_aligned = historical_vol.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(vol_aligned.values)

def historical_volatility_10_1_hour(df):
    """
    Calculate 10-period historical volatility on 1-hour resampled close prices.
    Historical volatility = standard deviation of log returns over 10 periods
    Expects df with columns: 'timestamp', 'close'
    Returns Series with volatility values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour closes
    close_1h = df['close'].resample('1h').last()
    
    # Calculate log returns
    log_returns = np.log(close_1h / close_1h.shift(1))
    
    # Calculate 10-period rolling standard deviation (historical volatility)
    historical_vol = log_returns.rolling(window=10).std()
    
    # Forward fill the volatility values
    historical_vol = historical_vol.ffill()
    
    # Reindex to original timestamps to maintain alignment
    vol_aligned = historical_vol.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(vol_aligned.values)

def stochastic_15_minute(df, k_period=14, d_period=3):
    """
    Calculate Stochastic oscillator %K on 15-minute resampled data.
    %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    Expects df with columns: 'timestamp', 'high', 'low', 'close'
    Returns Series with %K values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute OHLC
    ohlc_15m = df[['open', 'high', 'low', 'close']].resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate Stochastic components
    lowest_low = ohlc_15m['low'].rolling(window=k_period).min()
    highest_high = ohlc_15m['high'].rolling(window=k_period).max()
    
    # Calculate %K
    stoch_k = ((ohlc_15m['close'] - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Forward fill the stochastic values
    stoch_k = stoch_k.ffill()
    
    # Reindex to original timestamps to maintain alignment
    stoch_aligned = stoch_k.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(stoch_aligned.values)

def stochastic_1_hour(df, k_period=14, d_period=3):
    """
    Calculate Stochastic oscillator %K on 1-hour resampled data.
    %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    Expects df with columns: 'timestamp', 'high', 'low', 'close'
    Returns Series with %K values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour OHLC
    ohlc_1h = df[['open', 'high', 'low', 'close']].resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate Stochastic components
    lowest_low = ohlc_1h['low'].rolling(window=k_period).min()
    highest_high = ohlc_1h['high'].rolling(window=k_period).max()
    
    # Calculate %K
    stoch_k = ((ohlc_1h['close'] - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Forward fill the stochastic values
    stoch_k = stoch_k.ffill()
    
    # Reindex to original timestamps to maintain alignment
    stoch_aligned = stoch_k.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(stoch_aligned.values)

def adx_14_15_minute(df, period=14):
    """
    Calculate ADX (Average Directional Index) on 15-minute resampled data.
    ADX measures trend strength regardless of direction.
    Expects df with columns: 'timestamp', 'high', 'low', 'close'
    Returns Series with ADX values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute OHLC
    ohlc_15m = df[['open', 'high', 'low', 'close']].resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate True Range components
    high_low = ohlc_15m['high'] - ohlc_15m['low']
    high_close_prev = np.abs(ohlc_15m['high'] - ohlc_15m['close'].shift(1))
    low_close_prev = np.abs(ohlc_15m['low'] - ohlc_15m['close'].shift(1))
    
    # True Range is the maximum of the three
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate directional movements
    plus_dm = ohlc_15m['high'].diff()
    minus_dm = -ohlc_15m['low'].diff()
    
    # Set negative movements to 0 for +DM, positive movements to 0 for -DM
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
    
    # Calculate smoothed TR, +DM, and -DM using Wilder's smoothing (EMA with alpha = 1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di_raw = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_di_raw = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_di_raw / atr)
    minus_di = 100 * (minus_di_raw / atr)
    
    # Calculate DX (Directional Index)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX using Wilder's smoothing
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    # Forward fill the ADX values
    adx = adx.ffill()
    
    # Reindex to original timestamps to maintain alignment
    adx_aligned = adx.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(adx_aligned.values)

def adx_14_1_hour(df, period=14):
    """
    Calculate ADX (Average Directional Index) on 1-hour resampled data.
    ADX measures trend strength regardless of direction.
    Expects df with columns: 'timestamp', 'high', 'low', 'close'
    Returns Series with ADX values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour OHLC
    ohlc_1h = df[['open', 'high', 'low', 'close']].resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate True Range components
    high_low = ohlc_1h['high'] - ohlc_1h['low']
    high_close_prev = np.abs(ohlc_1h['high'] - ohlc_1h['close'].shift(1))
    low_close_prev = np.abs(ohlc_1h['low'] - ohlc_1h['close'].shift(1))
    
    # True Range is the maximum of the three
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate directional movements
    plus_dm = ohlc_1h['high'].diff()
    minus_dm = -ohlc_1h['low'].diff()
    
    # Set negative movements to 0 for +DM, positive movements to 0 for -DM
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
    
    # Calculate smoothed TR, +DM, and -DM using Wilder's smoothing (EMA with alpha = 1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di_raw = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_di_raw = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_di_raw / atr)
    minus_di = 100 * (minus_di_raw / atr)
    
    # Calculate DX (Directional Index)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX using Wilder's smoothing
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    # Forward fill the ADX values
    adx = adx.ffill()
    
    # Reindex to original timestamps to maintain alignment
    adx_aligned = adx.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(adx_aligned.values)

def donchian_distance_15_minute(df, period=20):
    """
    Calculate distance from Donchian Channel middle line on 15-minute resampled data.
    Middle Line = (Highest High + Lowest Low) / 2 over period
    Distance = ((Close - Middle Line) / Middle Line) * 100
    Expects df with columns: 'timestamp', 'high', 'low', 'close'
    Returns Series with percentage distance values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 15-minute OHLC
    ohlc_15m = df[['open', 'high', 'low', 'close']].resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate Donchian Channel components
    donchian_high = ohlc_15m['high'].rolling(window=period).max()
    donchian_low = ohlc_15m['low'].rolling(window=period).min()
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Calculate percentage distance from Donchian middle line
    distance_pct = ((ohlc_15m['close'] - donchian_middle) / donchian_middle) * 100
    
    # Forward fill the distance values to propagate them to all timestamps
    distance_pct = distance_pct.ffill()
    
    # Reindex to original timestamps to maintain alignment
    distance_aligned = distance_pct.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(distance_aligned.values)

def ichimoku_base_line_1_hour(df, period=26):
    """
    Calculate Ichimoku Base Line (Kijun-sen) on 1-hour resampled data.
    Base Line = (Highest High + Lowest Low) / 2 over period
    Expects df with columns: 'timestamp', 'high', 'low'
    Returns Series with Base Line values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour OHLC
    ohlc_1h = df[['open', 'high', 'low', 'close']].resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate Ichimoku Base Line (Kijun-sen)
    highest_high = ohlc_1h['high'].rolling(window=period).max()
    lowest_low = ohlc_1h['low'].rolling(window=period).min()
    base_line = (highest_high + lowest_low) / 2
    
    # Forward fill the Base Line values to propagate them to all timestamps
    base_line = base_line.ffill()
    
    # Reindex to original timestamps to maintain alignment
    base_line_aligned = base_line.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(base_line_aligned.values)

def vwap_above_below_1_hour(df):
    """
    Calculate binary indicator for price above/below 1-hour VWAP.
    Returns 1 if close is above VWAP, 0 if below VWAP.
    Expects df with columns: 'timestamp', 'close', 'volume'
    Returns Series with binary values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Check if volume column exists, if not create dummy volume
    if 'volume' not in df.columns:
        df['volume'] = 1  # Equal weight if no volume data
    
    # Resample to 1-hour OHLCV
    ohlcv_1h = df[['open', 'high', 'low', 'close', 'volume']].resample('1h').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Calculate typical price for VWAP calculation
    ohlcv_1h['typical_price'] = (ohlcv_1h['high'] + ohlcv_1h['low'] + ohlcv_1h['close']) / 3
    
    # Calculate VWAP for each 1-hour session
    ohlcv_1h['price_volume'] = ohlcv_1h['typical_price'] * ohlcv_1h['volume']
    ohlcv_1h['vwap'] = ohlcv_1h['price_volume'] / ohlcv_1h['volume']
    
    # Create binary indicator: 1 if close > VWAP, 0 if close <= VWAP
    ohlcv_1h['above_vwap'] = (ohlcv_1h['close'] > ohlcv_1h['vwap']).astype(int)
    
    # Forward fill the binary indicator values
    vwap_indicator = ohlcv_1h['above_vwap'].ffill()
    
    # Reindex to original timestamps to maintain alignment
    indicator_aligned = vwap_indicator.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(indicator_aligned.values)

def rate_of_change_1_hour(df, period=10):
    """
    Calculate Rate of Change (ROC) on 1-hour resampled close prices.
    ROC = ((Current Price - Price N periods ago) / Price N periods ago) * 100
    Expects df with columns: 'timestamp', 'close'
    Returns Series with ROC values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour closes
    close_1h = df['close'].resample('1h').last()
    
    # Calculate Rate of Change
    roc = ((close_1h - close_1h.shift(period)) / close_1h.shift(period)) * 100
    
    # Forward fill the ROC values to propagate them to all timestamps
    roc = roc.ffill()
    
    # Reindex to original timestamps to maintain alignment
    roc_aligned = roc.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(roc_aligned.values)

def bollinger_band_width_1_hour(df, period=20, num_std=2):
    """
    Calculate Bollinger Band width on 1-hour resampled close prices.
    Width = (Upper Band - Lower Band) / Middle Band * 100
    Expects df with columns: 'timestamp', 'close'
    Returns Series with BB width values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour closes
    close_1h = df['close'].resample('1h').last()
    
    # Calculate Bollinger Bands components
    sma = close_1h.rolling(window=period).mean()  # Middle Band
    std = close_1h.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    # Calculate Bollinger Band width as percentage
    bb_width = ((upper_band - lower_band) / sma) * 100
    
    # Forward fill the width values to propagate them to all timestamps
    bb_width = bb_width.ffill()
    
    # Reindex to original timestamps to maintain alignment
    bb_width_aligned = bb_width.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(bb_width_aligned.values)

def distance_from_sma_20_1_hour(df):
    """
    Calculate distance from 20-period SMA on 1-hour resampled close prices.
    Expects df with columns: 'timestamp', 'close'
    Returns Series with percentage distance values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour closes
    close_1h = df['close'].resample('1h').last()
    
    # Calculate 20-period SMA
    sma_20 = close_1h.rolling(window=20).mean()
    
    # Calculate percentage distance from SMA
    distance_pct = ((close_1h - sma_20) / sma_20) * 100
    
    # Forward fill the distance values to propagate them to all timestamps
    distance_pct = distance_pct.ffill()
    
    # Reindex to original timestamps to maintain alignment
    distance_aligned = distance_pct.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(distance_aligned.values)

def sma_5_slope_1_hour(df):
    """
    Calculate the slope of 5-period SMA on 1-hour resampled close prices.
    Expects df with columns: 'timestamp', 'close'
    Returns Series with slope values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour closes
    close_1h = df['close'].resample('1h').last()
    
    # Calculate 5-period SMA
    sma_5 = close_1h.rolling(window=5).mean()
    
    # Calculate slope (rate of change) of SMA
    # Slope = change in SMA / change in time (1 period = 1)
    slope = sma_5.diff()
    
    # Forward fill the slope values to propagate them to all timestamps
    slope = slope.ffill()
    
    # Reindex to original timestamps to maintain alignment
    slope_aligned = slope.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(slope_aligned.values)

def atr_1_hour(df, period=14):
    """
    Calculate ATR on 1-hour resampled OHLC data.
    Expects df with columns: 'timestamp', 'open', 'high', 'low', 'close'
    Returns Series with ATR values aligned to input timestamps
    """
    # Copy and ensure datetime index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 1-hour OHLC
    ohlc_1h = df[['open', 'high', 'low', 'close']].resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Calculate True Range components
    ohlc_1h['previous_close'] = ohlc_1h['close'].shift(1)
    ohlc_1h['tr1'] = ohlc_1h['high'] - ohlc_1h['low']
    ohlc_1h['tr2'] = (ohlc_1h['high'] - ohlc_1h['previous_close']).abs()
    ohlc_1h['tr3'] = (ohlc_1h['low'] - ohlc_1h['previous_close']).abs()
    
    # Calculate True Range as max of the three components
    ohlc_1h['true_range'] = ohlc_1h[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR as rolling mean of True Range
    atr = ohlc_1h['true_range'].rolling(window=period).mean()
    
    # Forward fill the ATR values to propagate them to all timestamps
    atr = atr.ffill()
    
    # Reindex to original timestamps to maintain alignment
    atr_aligned = atr.reindex(df.index)
    
    # Return as a Series with numeric index matching the original DataFrame's row order
    return pd.Series(atr_aligned.values)

def calculate_adx_14(df):
    df = df.copy()
    df = df.sort_values("timestamp")

    df["up_move"] = df["high"] - df["high"].shift(1)
    df["down_move"] = df["low"].shift(1) - df["low"]

    df["+DM"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0.0)
    df["-DM"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0.0)

    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = (df["high"] - df["close"].shift(1)).abs()
    df["tr3"] = (df["low"] - df["close"].shift(1)).abs()
    df["TR"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

    df["ATR"] = df["TR"].rolling(window=14).mean()
    df["+DI"] = 100 * (df["+DM"].rolling(window=14).mean() / df["ATR"])
    df["-DI"] = 100 * (df["-DM"].rolling(window=14).mean() / df["ATR"])

    df["DX"] = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])) * 100
    df["adx_14"] = df["DX"].rolling(window=14).mean()

    return df["adx_14"]

def donchian_distance_20(df):
    df = df.copy()
    df = df.sort_values("timestamp")

    df["donchian_high"] = df["high"].rolling(window=20).max()
    df["donchian_low"] = df["low"].rolling(window=20).min()

    df["donchian_mean"] = (df["donchian_high"] + df["donchian_low"]) / 2

    df["donchian_distance_20"] = df["close"] - df["donchian_mean"]

    return df["donchian_distance_20"]

def donchian_20_above_streak(df):
    df = df.copy()
    df = df.sort_values("timestamp")

    df["donchian_high"] = df["high"].rolling(window=20).max()
    df["donchian_low"] = df["low"].rolling(window=20).min()
    df["donchian_mean"] = (df["donchian_high"] + df["donchian_low"]) / 2

    condition = df["close"] > df["donchian_mean"]

    streak = []
    count = 0
    for is_above in condition:
        if is_above:
            count += 1
        else:
            count = 0
        streak.append(count)

    df["donchian_20_above_streak"] = streak

    return df["donchian_20_above_streak"]

def donchian_20_below_streak(df):
    df = df.copy()
    df = df.sort_values("timestamp")

    df["donchian_high"] = df["high"].rolling(window=20).max()
    df["donchian_low"] = df["low"].rolling(window=20).min()
    df["donchian_mean"] = (df["donchian_high"] + df["donchian_low"]) / 2

    condition = df["close"] < df["donchian_mean"]

    streak = []
    count = 0
    for is_below in condition:
        if is_below:
            count += 1
        else:
            count = 0
        streak.append(count)

    df["donchian_20_below_streak"] = streak

    return df["donchian_20_below_streak"]

def calculate_30_day_volatility(df):
    df = df.copy()
    df = standardize_column(df)
    df = df.sort_values("timestamp")

    df["returns"] = df["value"].pct_change()

    df["30d_volatility"] = df["returns"].rolling(window=30).std()

    return df["30d_volatility"]

def calculate_30_day_moving_average(df):
    df = df.copy()   
    df = standardize_column(df)
    df = df.sort_values("timestamp")
    
    df["30d_moving_avg"] = df["value"].rolling(window=30).mean()
    
    return df["30d_moving_avg"]

def return_difference_1y(df):
    """
    Calculate the difference in 1-year returns between two assets.
    Requires exactly 2 features with close price data (close and close_1).
    Formula: (Asset1_1y_return - Asset2_1y_return)
    """
    df = df.copy()
    df = df.sort_values("timestamp")
    
    # Check for required columns
    if "close" not in df.columns or "close_1" not in df.columns:
        raise ValueError("Function requires 'close' and 'close_1' columns")
    
    asset1_col = "close"
    asset2_col = "close_1"
    
    # Calculate 1-year (252 trading days) returns for both assets
    df[f"{asset1_col}_1y_return"] = df[asset1_col].pct_change(periods=252) * 100
    df[f"{asset2_col}_1y_return"] = df[asset2_col].pct_change(periods=252) * 100
    
    # Calculate the difference in returns (Asset1 - Asset2)
    df["return_difference_1y"] = df[f"{asset1_col}_1y_return"] - df[f"{asset2_col}_1y_return"]
    
    # Drop rows with NaN values (first 252 rows won't have 1-year data)
    result = df["return_difference_1y"].dropna()
    
    return result

export_functions = {
    "rsi" : rsi,
    "yoy" : yoy,
    "raw" : raw,
    "return_30d" : return_30d,
    "return_1d" : return_1d,
    "2_asset_correlation" : correlation_2_assets,
    "ytd" : return_ytd,
    "distance_from_sma_200" : distance_from_sma_200,
    "atr_14": calculate_atr_14,
    "adx_14" : calculate_adx_14,
    "donchian_distance_20" : donchian_distance_20,
    "donchian_20_above_streak" : donchian_20_above_streak,
    "donchian_20_below_streak" : donchian_20_below_streak,
    "calculate_30_day_volatility" : calculate_30_day_volatility,
    "calculate_30_day_moving_average" : calculate_30_day_moving_average,
    "return_difference_1y" : return_difference_1y,
    "rsi_15_minute": rsi_15_minute,
    "rsi_1_hour": rsi_1_hour,
    "atr_15_minute": atr_15_minute,
    "atr_1_hour": atr_1_hour,
    "sma_5_slope_15_minute": sma_5_slope_15_minute,
    "sma_5_slope_1_hour": sma_5_slope_1_hour,
    "distance_from_sma_5_15_minute": distance_from_sma_5_15_minute,
    "distance_from_sma_5_1_hour": distance_from_sma_5_1_hour,
    "distance_from_sma_20_15_minute": distance_from_sma_20_15_minute,
    "distance_from_sma_20_1_hour": distance_from_sma_20_1_hour,
    "bollinger_band_width_15_minute": bollinger_band_width_15_minute,
    "bollinger_band_width_1_hour": bollinger_band_width_1_hour,
    "rate_of_change_15_minute": rate_of_change_15_minute,
    "rate_of_change_1_hour": rate_of_change_1_hour,
    "vwap_above_below_1_hour": vwap_above_below_1_hour,
    "ichimoku_base_line_15_minute": ichimoku_base_line_15_minute,
    "ichimoku_base_line_1_hour": ichimoku_base_line_1_hour,
    "macd_histogram_15_minute": macd_histogram_15_minute,
    "macd_histogram_1_hour": macd_histogram_1_hour,
    "historical_volatility_10_15_minute": historical_volatility_10_15_minute,
    "historical_volatility_10_1_hour": historical_volatility_10_1_hour,
    "stochastic_15_minute": stochastic_15_minute,
    "stochastic_1_hour": stochastic_1_hour,
    "adx_14_15_minute": adx_14_15_minute,
    "adx_14_1_hour": adx_14_1_hour,
    "donchian_distance_15_minute": donchian_distance_15_minute
}

# Function input requirements mapping
function_requirements = {
    "rsi": {
        "required_columns": ["close"],
        "description": "Relative Strength Index - requires close price data",
        "min_features": 1
    },
    "rsi_15_minute": {
        "required_columns": ["close"],
        "description": "15-minute RSI - requires close price data",
        "min_features": 1
    },
    "rsi_1_hour": {
        "required_columns": ["close"],
        "description": "1-hour RSI - requires close price data",
        "min_features": 1
    },
    "yoy": {
        "required_columns": ["close"],
        "description": "Year-over-Year percentage change - requires close price or value data", 
        "min_features": 1
    },
    "raw": {
        "required_columns": ["any"],
        "description": "Raw data passthrough - accepts any column(s)",
        "min_features": 1
    },
    "return_30d": {
        "required_columns": ["close"],
        "description": "30-day return calculation - requires close price data",
        "min_features": 1
    },
    "return_1d": {
        "required_columns": ["close"],
        "description": "1-day return calculation - requires close price data",
        "min_features": 1
    },
    "2_asset_correlation": {
        "required_columns": ["close", "close_1"],
        "description": "Correlation between two assets - requires close price from 2 different assets",
        "min_features": 2
    },
    "ytd": {
        "required_columns": ["close"],
        "description": "Year-to-date return - requires close price data",
        "min_features": 1
    },
    "distance_from_sma_200": {
        "required_columns": ["close"],
        "description": "Distance from 200-day Simple Moving Average - requires close price data",
        "min_features": 1
    },
    "atr_14": {
        "required_columns": ["high", "low", "close"],
        "description": "14-day Average True Range - requires high, low, close price data",
        "min_features": 1
    },
    "atr_15_minute": {
        "required_columns": ["open", "high", "low", "close"],
        "description": "15-minute ATR - requires open, high, low, close price data",
        "min_features": 1
    },
    "atr_1_hour": {
        "required_columns": ["open", "high", "low", "close"],
        "description": "1-hour ATR - requires open, high, low, close price data",
        "min_features": 1
    },
    "sma_5_slope_15_minute": {
        "required_columns": ["close"],
        "description": "15-minute SMA 5 slope - requires close price data",
        "min_features": 1
    },
    "sma_5_slope_1_hour": {
        "required_columns": ["close"],
        "description": "1-hour SMA 5 slope - requires close price data",
        "min_features": 1
    },
    "distance_from_sma_5_15_minute": {
        "required_columns": ["close"],
        "description": "15-minute distance from SMA 5 - requires close price data",
        "min_features": 1
    },
    "distance_from_sma_5_1_hour": {
        "required_columns": ["close"],
        "description": "1-hour distance from SMA 5 - requires close price data",
        "min_features": 1
    },
    "distance_from_sma_20_15_minute": {
        "required_columns": ["close"],
        "description": "15-minute distance from SMA 20 - requires close price data",
        "min_features": 1
    },
    "distance_from_sma_20_1_hour": {
        "required_columns": ["close"],
        "description": "1-hour distance from SMA 20 - requires close price data",
        "min_features": 1
    },
    "bollinger_band_width_15_minute": {
        "required_columns": ["close"],
        "description": "15-minute Bollinger Band width - requires close price data",
        "min_features": 1
    },
    "bollinger_band_width_1_hour": {
        "required_columns": ["close"],
        "description": "1-hour Bollinger Band width - requires close price data",
        "min_features": 1
    },
    "rate_of_change_15_minute": {
        "required_columns": ["close"],
        "description": "15-minute Rate of Change - requires close price data",
        "min_features": 1
    },
    "rate_of_change_1_hour": {
        "required_columns": ["close"],
        "description": "1-hour Rate of Change - requires close price data",
        "min_features": 1
    },
    "vwap_above_below_1_hour": {
        "required_columns": ["open", "high", "low", "close", "volume"],
        "description": "1-hour VWAP above/below binary indicator - requires OHLCV data",
        "min_features": 1
    },
    "ichimoku_base_line_15_minute": {
        "required_columns": ["open", "high", "low", "close"],
        "description": "15-minute Ichimoku Base Line - requires OHLC data",
        "min_features": 1
    },
    "ichimoku_base_line_1_hour": {
        "required_columns": ["open", "high", "low", "close"],
        "description": "1-hour Ichimoku Base Line - requires OHLC data",
        "min_features": 1
    },
    "macd_histogram_15_minute": {
        "required_columns": ["close"],
        "description": "15-minute MACD histogram level - requires close price data",
        "min_features": 1
    },
    "macd_histogram_1_hour": {
        "required_columns": ["close"],
        "description": "1-hour MACD histogram level - requires close price data",
        "min_features": 1
    },
    "historical_volatility_10_15_minute": {
        "required_columns": ["close"],
        "description": "10-period historical volatility on 15-minute data - requires close price data",
        "min_features": 1
    },
    "historical_volatility_10_1_hour": {
        "required_columns": ["close"],
        "description": "10-period historical volatility on 1-hour data - requires close price data",
        "min_features": 1
    },
    "stochastic_15_minute": {
        "required_columns": ["open", "high", "low", "close"],
        "description": "15-minute Stochastic oscillator %K - requires OHLC data",
        "min_features": 1
    },
    "stochastic_1_hour": {
        "required_columns": ["open", "high", "low", "close"],
        "description": "1-hour Stochastic oscillator %K - requires OHLC data",
        "min_features": 1
    },
    "adx_14_15_minute": {
        "required_columns": ["open", "high", "low", "close"],
        "description": "15-minute ADX 14 - requires OHLC data",
        "min_features": 1
    },
    "adx_14_1_hour": {
        "required_columns": ["open", "high", "low", "close"],
        "description": "1-hour ADX 14 - requires OHLC data",
        "min_features": 1
    },
    "donchian_distance_15_minute": {
        "required_columns": ["open", "high", "low", "close"],
        "description": "15-minute distance from Donchian Channel middle line - requires OHLC data",
        "min_features": 1
    },
    "adx_14": {
        "required_columns": ["high", "low", "close"],
        "description": "14-day Average Directional Index - requires high, low, close price data",
        "min_features": 1
    },
    "donchian_distance_20": {
        "required_columns": ["high", "low", "close"],
        "description": "Distance from 20-day Donchian Channel midpoint - requires high, low, close",
        "min_features": 1
    },
    "donchian_20_above_streak": {
        "required_columns": ["high", "low", "close"],
        "description": "Consecutive days above Donchian Channel - requires high, low, close",
        "min_features": 1
    },
    "donchian_20_below_streak": {
        "required_columns": ["high", "low", "close"],
        "description": "Consecutive days below Donchian Channel - requires high, low, close",
        "min_features": 1
    },
    "calculate_30_day_volatility": {
        "required_columns": ["value", "close", "open", "high", "low", "volume"],
        "description": "30-day rolling volatility - accepts any price/value column",
        "min_features": 1
    },
    "calculate_30_day_moving_average": {
        "required_columns": ["value", "close", "open", "high", "low", "volume"],
        "description": "30-day moving average - accepts any price/value column",
        "min_features": 1
    },
    "return_difference_1y": {
        "required_columns": ["close"],
        "description": "1-year return difference between two assets - requires close price from exactly 2 assets",
        "min_features": 2
    }
}