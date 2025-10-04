import pandas as pd
import yfinance as yf
import time
from typing import List, Dict
from .tools import get_tqdm


def download_data(symbols: List[str], period: str, interval: str, batch_size: int = 100, delay: float = 0.1) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for multiple symbols using Yahoo Finance.

    Args:
        symbols (List[str]): List of ticker symbols to download.
        period (str): Data period (e.g., "1d", "5d", "1mo", "1y").
        interval (str): Candlestick interval (e.g., "1m", "15m", "1h", "1d").
        batch_size (int, optional): Number of symbols per request batch. Defaults to 100.
        delay (float, optional): Delay (in seconds) between batches to avoid rate limiting. Defaults to 0.1.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of symbol → OHLCV DataFrame.
            Each DataFrame contains: Datetime, Open, High, Low, Close, Volume, Symbol.
    """
    tqdm = get_tqdm()

    results: Dict[str, pd.DataFrame] = {}

    num_batches = (len(symbols) - 1) // batch_size + 1
    for i in tqdm(range(0, len(symbols), batch_size), desc="Downloading batches", total=num_batches):
        batch = symbols[i:i + batch_size]

        try:
            raw_df = yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
                threads=True,
                ignore_tz=True,
                auto_adjust=False
            )
        except Exception as e:
            print(f"Error downloading batch {batch}: {e}")
            time.sleep(delay)
            continue

        if raw_df.empty:
            time.sleep(delay)
            continue

        # Normalize structure
        raw_df = raw_df.reset_index(drop=False)

        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = [
                f"{symbol}_{field}" if field else "Datetime"
                for symbol, field in raw_df.columns
            ]
        else:
            raw_df.columns = [
                "Datetime" if col == "Date" else f"{batch[0]}_{col}"
                for col in raw_df.columns
            ]

        # Extract data per symbol
        for symbol in batch:
            cols = [col for col in raw_df.columns if col.startswith(f"{symbol}_")]
            if not cols:
                continue

            df_symbol = raw_df[["Datetime"] + cols].copy()
            df_symbol.columns = [
                col.split("_", 1)[1] if "_" in col else col
                for col in df_symbol.columns
            ]
            df_symbol["Symbol"] = symbol

            if df_symbol.empty or df_symbol["Open"].isna().all():
                continue

            results[symbol] = df_symbol

        time.sleep(delay)

    return results

def write_list_to_file(data: List[str], filepath: str) -> None:
    """
    Write a list of strings to a file as comma-separated values.

    Args:
        data (List[str]): List of strings to write.
        filepath (str): Path to the output file.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(",".join(data))
    except Exception as e:
        raise IOError(f"Failed to write to file {filepath}: {e}")

def read_list_from_file(filepath: str) -> List[str]:
    """
    Read a comma-separated list of strings from a file.

    Args:
        filepath (str): Path to the input file.

    Returns:
        List[str]: List of strings read from the file.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read().strip()
            return content.split(",") if content else []
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise IOError(f"Failed to read from file {filepath}: {e}")
    