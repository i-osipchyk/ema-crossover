import pandas as pd
from typing import List, Dict
import glob
import os
from datetime import datetime


def find_potential_entries(
    stock_data_labeled: Dict[str, pd.DataFrame],
    short_ema: int = 8,
    long_ema: int = 20,
    min_crossover_vol: int = 500_000,
    min_price: float = 20.0,
    max_atr: float = 5.77,
    shift: int = 0,
    filter_set: List[str] = [
        "Above_EMA_34",
        "Above_EMA_50",
        "Above_EMA_200",
        "MACD_Positive",
        "MACD_Signal_Negative",
        "Volume",
        "Price",
        "ATR"
    ],
) -> List[str]:
    """
    Scan all stocks and return list of symbols that may have a long entry for a given session.

    Args:
        stock_data_labeled (dict): Dict of {symbol: DataFrame} with indicators and crossovers.
        short_ema (int): Short EMA period.
        long_ema (int): Long EMA period.
        min_crossover_vol (int): Minimum volume on crossover day.
        min_price (float): Minimum entry price.
        max_atr (float): Maximum ATR% allowed.
        shift (int): Number of rows to shift backwards (0 = last row).
        filters_set (list of str): List of filters to apply. 
            Available: ["Above_EMA_34", "Above_EMA_50", "Above_EMA_200", 
                        "MACD_Positive", "MACD_Signal_Negative", 
                        "Volume", "Price", "ATR"]

    Returns:
        List[str]: Symbols that may have entry setups.
    """
    if filter_set is None:
        filter_set = ["Above_EMA_34", "Above_EMA_50", "Above_EMA_200",
                   "MACD_Positive", "MACD_Signal_Negative",
                   "Volume", "Price", "ATR"]

    candidates = []
    crossover_col = f"EMA_Close_{short_ema}_EMA_Close_{long_ema}_Crossover"

    for symbol, df in stock_data_labeled.items():
        if df.empty or shift >= len(df):
            continue

        last_row = df.iloc[-1 - shift]

        if crossover_col not in df.columns or last_row[crossover_col] != "Bullish":
            continue

        passed = True

        if "Above_EMA_34" in filter_set and not last_row.get("Above_EMA_Close_34", False):
            passed = False
        if "Above_EMA_50" in filter_set and not last_row.get("Above_EMA_Close_50", False):
            passed = False
        if "Above_EMA_200" in filter_set and not last_row.get("Above_EMA_Close_200", False):
            passed = False
        if "MACD_Positive" in filter_set and last_row.get("MACD", 0) <= 0:
            passed = False
        if "MACD_Signal_Negative" in filter_set and last_row.get("MACD_Signal", 0) >= 0:
            passed = False
        if "Volume" in filter_set and last_row.get("Volume", 0) < min_crossover_vol:
            passed = False
        if "Price" in filter_set and last_row.get("Close", 0) < min_price:
            passed = False
        if "ATR" in filter_set and last_row.get("ATR%", 100) > max_atr:
            passed = False

        if passed:
            candidates.append(symbol)

    return candidates, filter_set

def get_latest_labeled_file(folder: str = "daily_labeled") -> str | None:
    """Return the path of the latest labeled file, or None if no file exists."""
    files = glob.glob(os.path.join(folder, "*.pkl"))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def market_data_is_fresh(latest_file: str) -> bool:
    """
    Check if market session has renewed since the latest file.
    For simplicity: compare the file's modification date with today's date (UTC).
    """
    if latest_file is None:
        return False

    file_date = datetime.fromtimestamp(os.path.getctime(latest_file)).date()
    today = datetime.utcnow().date()
    return file_date == today
