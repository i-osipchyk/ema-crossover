import pandas as pd
from typing import List, Dict, Callable, Tuple, Any
from .tools import get_tqdm


def apply_to_dict(stock_dict: Dict[str, pd.DataFrame], function: Callable[..., pd.DataFrame], **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Apply a transformation function to each DataFrame in a dictionary.

    Args:
        stock_dict (Dict[str, pd.DataFrame]): Mapping of symbol → DataFrame.
        function (Callable[..., pd.DataFrame]): Function to apply to each DataFrame.
        **kwargs: Additional keyword arguments for the function.

    Returns:
        Dict[str, pd.DataFrame]: Updated mapping with transformed DataFrames.
    """
    tqdm = get_tqdm()

    new_stock_dict: Dict[str, pd.DataFrame] = {}
    for symbol, df in tqdm(stock_dict.items(), desc="Processing symbols", total=len(stock_dict)):
        df_copy = df.copy()
        try:
            new_stock_dict[symbol] = function(df_copy, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error processing {symbol}: {e}")
    return new_stock_dict


def calculate_ma(df: pd.DataFrame, period: int, source: str, method: str) -> pd.DataFrame:
    """
    Calculate a moving average (EMA or SMA) and add it as a new column.

    Args:
        df (pd.DataFrame): DataFrame containing a source column (e.g., 'Close').
        period (int): Lookback period for the moving average.
        source (str, optional): Column to calculate MA on. Defaults to "Close".
        method (str, optional): Type of moving average ("EMA" or "SMA"). Defaults to "EMA".

    Returns:
        pd.DataFrame: DataFrame with the new MA column added.

    Raises:
        ValueError: If the source column is missing, if period <= 0, or if method is invalid.
        TypeError: If period is not an integer.
    """
    try:
        if method.upper() == "EMA":
            ma_series = df[source].ewm(span=period, adjust=False).mean()
            ma_series.iloc[:period - 1] = pd.NA
        else:  # SMA
            ma_series = df[source].rolling(window=period).mean()

        col_name = f"{method.upper()}_{source}_{period}"
        df[col_name] = ma_series
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to calculate {method.upper()} on column '{source}' with period {period}: {e}")


def calculate_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR) and ATR%.

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.
        period (int, optional): ATR lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: DataFrame with added 'ATR_{period}' and 'ATR%' columns.
    """
    try:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr_col = f"ATR_{period}"
        df[atr_col] = tr.rolling(period).mean()
        df["ATR%"] = df[atr_col] / df["Close"] * 100
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to calculate ATR: {e}") 


def calculate_macd(df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int, source_col: str) -> pd.DataFrame:
    """
    Calculate MACD, Signal, and Histogram with configurable periods.

    Args:
        df (pd.DataFrame): DataFrame with a column to calculate MACD on (default 'Close').
        fast_period (int, optional): Period for fast EMA. Defaults to 12.
        slow_period (int, optional): Period for slow EMA. Defaults to 26.
        signal_period (int, optional): Period for signal EMA. Defaults to 9.
        source_col (str, optional): Column name to calculate MACD from. Defaults to 'Close'.

    Returns:
        pd.DataFrame: DataFrame with added 'MACD', 'MACD_Signal', 'MACD_Hist' columns.
    """
    try:
        ema_fast = df[source_col].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[source_col].ewm(span=slow_period, adjust=False).mean()

        df["MACD"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        return df
    except Exception as e:
        raise RuntimeError(f"Failed to calculate MACD: {e}")


def label_candle_color(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each candlestick as Green, Red, or Doji.

    Args:
        df (pd.DataFrame): DataFrame with 'Open' and 'Close' columns.

    Returns:
        pd.DataFrame: DataFrame with added 'Candle_Color' column.
    """
    try:
        df["Candle_Color"] = df.apply(
            lambda row: (
                "Green" if row["Close"] > row["Open"]
                else "Red" if row["Close"] < row["Open"]
                else "Doji"
            ),
            axis=1
        )
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to label candle color: {e}") 


def mark_crossovers(df: pd.DataFrame, short_ma_params: Tuple, long_ma_params: Tuple) -> pd.DataFrame:
    """
    Detect moving average crossovers (Bullish or Bearish).

    Args:
        df (pd.DataFrame): DataFrame containing the required MA columns.
        short_ma_params (Tuple[int, str, str]): (period, source, method) for the short MA.
        long_ma_params (Tuple[int, str, str]): (period, source, method) for the long MA.

    Returns:
        pd.DataFrame: DataFrame with an added crossover signal column.

    Raises:
        ValueError: If the required MA columns are missing or periods are invalid.
    """
    try:
        short_p, short_s, short_m = short_ma_params
        long_p, long_s, long_m = long_ma_params

        if short_p >= long_p:
            raise ValueError(
                f"Short MA must have smaller period than Long MA.\nFound Short MA Period: {short_p}, Long MA Period: {long_p}"
            )

        short_col = f"{short_m}_{short_s}_{short_p}"
        long_col = f"{long_m}_{long_s}_{long_p}"
        crossover_col = f"{short_col}_{long_col}_Crossover"

        if short_col not in df.columns or long_col not in df.columns:
            raise ValueError(
                f"Required columns '{short_col}' and/or '{long_col}' not found in DataFrame."
            )

        cross_up = (df[short_col] > df[long_col]) & (df[short_col].shift(1) <= df[long_col].shift(1))
        cross_down = (df[short_col] < df[long_col]) & (df[short_col].shift(1) >= df[long_col].shift(1))

        df[crossover_col] = "No"
        df.loc[cross_up, crossover_col] = "Bullish"
        df.loc[cross_down, crossover_col] = "Bearish"

        return df
    except Exception as e:
        raise RuntimeError(f"Failed to calculate MA Crossover: {e}")  


def calculate_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the gap between the current Open and previous Close.

    Args:
        df (pd.DataFrame): DataFrame with 'Open' and 'Close' columns.
    
    Returns:
        pd.DataFrame: DataFrame with added 'Gap' and 'Gap%' columns.
    """
    try:
        df["Gap"] = df["Open"] - df["Close"].shift(1)
        df["Gap%"] = df["Gap"] / df["Close"].shift(1) * 100
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to calculate Gap: {e}")


def calculate_percentage_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage change between consecutive 'Close' prices.

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'Pct_Change' column.
    """
    try:
        df["Chg%"] = (df["Close"] - df['Open']) / df['Open'] * 100
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to calculate Percentage Change: {e}")


def calculate_rsi(df: pd.DataFrame, period: int = 14, source_col: str = "Close") -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI) and add it to the DataFrame.

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions in the price of a stock.

    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = (average gain / average loss)

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data containing a source column (e.g., "Close").
        period (int): Lookback period for RSI calculation. Default is 14.
        source_col (str): Column to calculate RSI from. Default is "Close".

    Returns:
        pd.DataFrame: DataFrame with a new column 'RSI_<period>' added.
    """
    if source_col not in df.columns:
        raise ValueError(f"Source column '{source_col}' not found in DataFrame.")

    delta = df[source_col].diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder’s smoothing method (EMA of gains/losses)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df[f"RSI_{period}"] = rsi.round(2)

    return df


def add_indicators(df: pd.DataFrame, indicators_config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Clean OHLCV data and compute technical indicators based on a config dictionary.

    Args:
        df (pd.DataFrame): Symbol DataFrame with OHLCV columns.
        indicators_config (Dict[str, Any], optional): Dictionary with indicator names as keys and parameters as values.

    Returns:
        pd.DataFrame: DataFrame with added indicators.
    """
    if indicators_config is None:
        raise ValueError("indicators_config must be provided as a dictionary.")

    # Ensure numeric OHLCV
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    # Moving averages
    for period, source, method in indicators_config.get("moving_averages", []):
        df = calculate_ma(df, period=period, source=source, method=method)
        ma_col = f"{method}_{source}_{period}"
        df[f"Above_{ma_col}"] = df["Close"] > df[ma_col]

    # Crossover signals
    for short_params, long_params in indicators_config.get("crossovers", []):
        df = mark_crossovers(df, short_params, long_params)

    # Other indicators
    if "atr" in indicators_config:
        atr_period = indicators_config.get("atr", {}).get("period", 14)
        df = calculate_atr(df, atr_period)
    if "macd" in indicators_config:
        macd_params = indicators_config.get("macd", {})
        fast_period = macd_params.get("fast_period", 12)
        slow_period = macd_params.get("slow_period", 26)
        signal_period = macd_params.get("signal_period", 9)
        source_col = macd_params.get("source_col", "Close")
        df = calculate_macd(df, fast_period, slow_period, signal_period, source_col)
    if "candle_color" in indicators_config:
        df = label_candle_color(df)
    if "gap" in indicators_config:
        df = calculate_gap(df)
    if "percentage_change" in indicators_config:
        df = calculate_percentage_change(df)
    if "rsi" in indicators_config:
        rsi_params = indicators_config.get("rsi", {})
        period = rsi_params.get("period", 14)
        source_col = rsi_params.get("source_col", "Close")
        df = calculate_rsi(df, period, source_col)

    return df
