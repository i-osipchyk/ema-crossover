import pandas as pd
from typing import List, Dict, Callable, Tuple
from tools import get_tqdm


def apply_to_dict(
    stock_dict: Dict[str, pd.DataFrame],
    function: Callable[..., pd.DataFrame],
    **kwargs
) -> Dict[str, pd.DataFrame]:
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

def calculate_ma(df: pd.DataFrame, period: int, source: str = "Close", method: str = "EMA") -> pd.DataFrame:
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

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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
    
def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD, Signal, and Histogram.

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' column.

    Returns:
        pd.DataFrame: DataFrame with added 'MACD', 'MACD_Signal', 'MACD_Hist' columns.
    """
    try:
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()

        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
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

def process_symbol_df(
    df: pd.DataFrame,
    ma_params: List[int] = [
        (8, "Close", "EMA"), (20, "Close", "EMA"), (34, "Close", "EMA"), (50, "Close", "EMA"), (200, "Close", "EMA")
    ],
    crossover_mas: List[Tuple[Tuple[int, str, str], Tuple[int, str, str]]] = [
        ((8, "Close", "EMA"), (20, "Close", "EMA"))
    ]
) -> pd.DataFrame:
    """
    Clean OHLCV data and compute multiple technical indicators.

    Args:
        df (pd.DataFrame): Symbol DataFrame with OHLCV columns.
        ma_params (List[Tuple[int, str, str]], optional): List of MA configs as (period, source, method).
        crossover_mas (List[Tuple[Tuple, Tuple]], optional): List of MA crossover pairs.
            Example: [((8, "Close", "EMA"), (20, "Close", "EMA"))]

    Returns:
        pd.DataFrame: DataFrame with added indicators.
    """
    # Ensure numeric OHLCV
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Round OHLC values
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    # --- Calculate EMAs, SMAs and above/below flags ---
    for period, source, method in ma_params:
        df = calculate_ma(df, period=period, source=source, method=method)
        ma_col = f"{method}_{source}_{period}"
        df[f"Above_{ma_col}"] = df["Close"] > df[ma_col]

    # --- Other indicators ---
    df = calculate_atr(df)
    df = calculate_macd(df)
    df = label_candle_color(df)

    # --- Add crossover signals ---
    for short_params, long_params in crossover_mas:
        df = mark_crossovers(df, short_params, long_params)

    return df
