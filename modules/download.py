import os
import time as timer
import re
import pytz
import pandas as pd
import yfinance as yf
import logging
from typing import List, Dict
from datetime import datetime, timedelta, time

from modules.tools import get_tqdm


logger = logging.getLogger()

NY_TZ = pytz.timezone("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

def get_latest_saved_file(data_dir: str) -> str:
    """
    Retrieve the most recently modified .pkl file in a directory.

    Args:
        data_dir (str): Path to the directory containing saved files.

    Returns:
        str: Full path to the latest saved file, or None if no files exist.
    """
    saved_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pkl")],
        key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
        reverse=True
    )
    return os.path.join(data_dir, saved_files[0]) if saved_files else None


def extract_timestamp(filename: str) -> datetime | None:
    """
    Extract a datetime object from a filename based on the pattern YYYY-MM-DD_HH-MM-SS.

    Args:
        filename (str): The filename containing the timestamp.

    Returns:
        datetime | None: Localized datetime in NY_TZ if extraction succeeds, otherwise None.
    """
    pattern = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    match = re.search(pattern, filename)
    if match:
        timestamp_str = match.group(1)
        try:
            file_dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
            return NY_TZ.localize(file_dt)
        except ValueError:
            return None
    return None


def is_market_hour(current_dt: datetime) -> bool:
    """
    Determine if a given datetime falls within U.S. market hours (9:30–16:00 ET, Mon–Fri).

    Args:
        current_dt (datetime): The datetime to check. Can be naive or timezone-aware.

    Returns:
        bool: True if within market hours, False otherwise.
    """
    if current_dt.tzinfo is None:
        current_dt = current_dt.replace(tzinfo=pytz.utc)

    current_dt_et = current_dt.astimezone(NY_TZ)

    if current_dt_et.weekday() >= 5:
        return False

    return MARKET_OPEN <= current_dt_et.time() <= MARKET_CLOSE


def find_closest_market_hour(current_dt: datetime) -> datetime:
    """
    Find the most recent market hour in the past relative to a given datetime.
    If executed during market hours, returns the current time.

    Args:
        current_dt (datetime): The reference datetime. Can be naive or timezone-aware.

    Returns:
        datetime: The closest market hour in NY_TZ.
    """
    if current_dt.tzinfo is None:
        current_dt = current_dt.replace(tzinfo=pytz.utc)

    current_dt_et = current_dt.astimezone(NY_TZ)

    # Return now for market hours
    if is_market_hour(current_dt):
        return current_dt_et

    # If after market close today
    if current_dt_et.weekday() < 5 and current_dt_et.time() > MARKET_CLOSE:
        return datetime.combine(current_dt_et.date(), MARKET_CLOSE, NY_TZ)

    # Otherwise, find last valid weekday close
    days_checked = 0
    while days_checked < 7:  # Safety stop
        current_dt_et -= timedelta(days=1)
        if current_dt_et.weekday() < 5:  # Monday–Friday
            return datetime.combine(current_dt_et.date(), MARKET_CLOSE, NY_TZ)
        days_checked += 1

    raise RuntimeError("Could not find a valid market day within the past week.")


def compare_timestamps(latest: datetime, now: datetime, interval: str) -> bool:
    """
    Compare two datetimes to determine if the difference is less than a specified interval.

    Args:
        latest (datetime): The previous timestamp.
        now (datetime): The current timestamp.
        interval (str): Interval string ('Xm', 'Xh', 'Xd') to compare.

    Returns:
        bool: True if the difference is less than the interval, False otherwise.
    """
    if latest is None:
        return False
    
    if interval.endswith("m"):
        delta = timedelta(minutes=int(interval[:-1]))
    elif interval.endswith("h"):
        delta = timedelta(hours=int(interval[:-1]))
    elif interval.endswith("d"):
        delta = timedelta(days=int(interval[:-1]))
    else:
        raise ValueError(f"Unsupported interval format: {interval}")
    
    return now - latest < delta


def check_symbols(existing_symbols: List[str], requested_symbols: List[str]) -> bool:
    """
    Check whether two lists of symbols are identical.
    """
    return set(existing_symbols) == set(requested_symbols)


def get_yf_data(symbols: List[str], period: str, interval: str, 
                batch_size: int = 100, delay: float = 0.1, add_tqdm: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Get OHLCV data for multiple symbols using Yahoo Finance.

    Args:
        symbols (List[str]): List of symbols to download.
        period (str): Data period (e.g., "1d", "5d", "1mo", "1y").
        interval (str): Candlestick interval (e.g., "1m", "15m", "1h", "1d").
        batch_size (int, optional): Number of symbols per request batch. Defaults to 100.
        delay (float, optional): Delay (in seconds) between batches to avoid rate limiting. Defaults to 0.1.
        add_tqdm (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of symbol → OHLCV DataFrame.
            Each DataFrame contains: Datetime, Open, High, Low, Close, Volume, Symbol.
    """
    if add_tqdm:
        tqdm = get_tqdm()
    else:
        tqdm = lambda x, **kwargs: x

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
            logger.error(f"Error downloading batch {batch}: {e}")
            timer.sleep(delay)
            continue

        if raw_df.empty:
            timer.sleep(delay)
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

            # Leaves all requested symbols in the final dataframe, so that the list can be compared in the future
            # if df_symbol.empty or df_symbol["Open"].isna().all():
            #     continue
            
            if df_symbol.empty:
                continue

            results[symbol] = df_symbol

        timer.sleep(delay)

    return results


def download_data(
        symbols: List[str], period: str, interval: str,
        batch_size: int = 100, delay: float = 0.1,
        expected_root: str = "/Users/ivanosipchyk/dev/investing/ema-crossover") -> Dict[str, pd.DataFrame]:
    
    # 1. Move to root dir
    current_dir = os.getcwd()
    
    if current_dir != expected_root:
        raise RuntimeError(
            f"Current working directory is {current_dir}, but expected {expected_root}. "
            "Please run this function from the project root."
        )
    
    data_dir = os.path.join(current_dir, "data", "historical_data", interval)
    os.makedirs(data_dir, exist_ok=True)

    # 2. Get lastest saved file for a given interval
    latest_file = get_latest_saved_file(data_dir)

    if latest_file:
        # 3. Compare file timestamps with the closest market hour
        latest_timestamp = extract_timestamp(latest_file)
        closest_market_hour = find_closest_market_hour(datetime.now(pytz.timezone("America/New_York")))
        if compare_timestamps(latest_timestamp, closest_market_hour, interval):

            # 4. Compare symbols
            latest_data = pd.read_pickle(latest_file)
            if check_symbols(list(latest_data.keys()), symbols):
                logger.info(f"Data is up-to-date as of {latest_timestamp} and contains same symbols. No download needed.")
                return latest_data

    historical_data = get_yf_data(symbols, period, interval, batch_size, delay)
   
    filename = f"historical_data_{datetime.now(NY_TZ).strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    save_path = os.path.join(data_dir, filename)

    pd.to_pickle(historical_data, save_path)
    logger.info(f"\n✅ Saved all downloaded data to {save_path}")

    return historical_data


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
    