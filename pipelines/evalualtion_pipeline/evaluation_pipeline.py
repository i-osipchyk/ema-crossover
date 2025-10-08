import os
from datetime import datetime, timedelta
from typing import List
import gspread
from google.oauth2.service_account import Credentials
import logging
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
import time

from modules.download import *
from modules.indicators import *
from modules.pipeline import *
from modules.tools import *
from modules.simulation import *
from modules.google_sheet_tools import *
from modules.google_sheets_manager import GoogleSheetsManager


# Remove existing handlers (important if something configured logging before)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(
    log_dir,
    f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)

# Configure logging to file + console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()  # use root logger

logger.info("Logger initialized successfully!")
logger.info(f"Writing logs to: {log_filename}")


DEFAULT_FILTER_SETS = [
    ["Above_EMA_34", "Above_EMA_50", "Above_EMA_200", "MACD_Positive", "Volume", "Price", "ATR"],
    ["Above_EMA_34", "Above_EMA_50", "Above_EMA_200", "MACD_Positive", "MACD_Signal_Negative", "Volume", "Price", "ATR"],
]


def main(
    sheet_url: str,
    filter_sets: List[List[str]],
    symbols_list: List[str] = None
) -> Dict[str, Any]:
    """
    Workflow:
        1. Connect to Google Sheets
        2. Download historical stock data and compute indicators.
        3. Identify potential trade entries using provided filter sets.
        4. Write potential entries to the "Potential Setups" tab.
        5. Read previous day entries and existing trade tabs.
        6. Filter stock data for previous day entries.
        7. Generate new trades for all trade parameter sets and update corresponding tabs.
        8. Evaluate trades in remaining tabs.
        9. Log results and return a summary dictionary.

    Args:
        sheet_url (str): Google Sheets URL.
        filters (List[str]): List of filter sets to apply for potential entries.
        symbols_list (List[str], optional): List of symbols. Defaults to a test list.
    """
    logger.info("Starting daily trading pipeline.\n")

    try:
        logger.info(f"Connecting to Google Sheets: {sheet_url}")
        gs_manager = GoogleSheetsManager(sheet_url=sheet_url)
        logger.info("Successfully connected to Google Sheets.\n")
    except Exception as e:
        logger.error(f"Could not connect to Google Sheets due to error: {e}")

    # 1. Download and label all data
    try:
        symbol_list_txt_path = "data/all-symbols-june-2025.txt"
        logger.info("Fetching symbol list from {symbol_list_txt_path}...")
        symbols_list = read_list_from_file(symbol_list_txt_path)

        logger.info(f"Downloading data for {len(symbols_list)} symbols...")

        stock_data = download_data(
            symbols=symbols_list,
            period="500d",
            interval="1d",
            batch_size=100,
        )

        logger.info(f"Adding indicators...")

        stock_data_with_indicators = apply_to_dict(stock_data, add_indicators)

        logger.info(f"Downloaded and added indicators for {len(stock_data_with_indicators)}.")
        logger.info(f"Example symbol: {list(stock_data_with_indicators.keys())[0]}")
        logger.info(f"Data columns: {list(next(iter(stock_data_with_indicators.values())).columns)}.\n")
    except Exception as e:
        logger.error(f"Could not download and add indicators to data due to error: {e}")

    # 2. Find potential entries based on different sets of filters
    try:
        logger.info(f"Finding potential entries for {len(filter_sets)} filter sets...")
        logger.debug(f"Filter sets: {filter_sets}")
        
        potential_entries_with_filters = [
            (entries, fset)
            for fset in filter_sets
            if (entries := find_potential_entries(stock_data_with_indicators, filter_set=fset)[0])
        ]

        logger.info(f"Found {len(potential_entries_with_filters)} potential entries sets.\n")
    except Exception as e:
        logger.error(f"Could not find potential entries due to error: {e}")

    # 3. Write potential entries and their filters
    try:
        logger.info(f"Writing potential entries and filters to Google Sheets...")

        latest_day = next(iter(stock_data_with_indicators.values()))['Datetime'].values[-1] # Get latest day
        
        gs_manager.write_potential_entries(
            latest_day=latest_day,
            potential_entries_with_filters=potential_entries_with_filters
        )

        logger.info("Successfully wrote potential entries.\n")
    except Exception as e:
        logger.error(f"Could not write potential entries due to error: {e}")

    # 4. Read all potential entries from the previous trading day and all existing trade tabs
    try:
        logger.info("Reading previous day setups...")

        prev_day_entries = gs_manager.read_previous_trading_day_setups()

        logger.info(f"Previous trading day setups: {prev_day_entries}.\n")
    except Exception as e:
        logger.error(f"Could not read previous day setups due to error: {e}")

    try:
        logger.info("Reading trade tabs setups...")

        trade_tabs = gs_manager.read_trade_tabs()

        logger.info(f"Found {len(trade_tabs)} trade tabs.\n")
    except Exception as e:
        logger.error(f"Could not read trade tabs due to error: {e}")

    # TODO: Select data for generation in trade generation function
    # Filter stock data for these entries
    stock_data_for_entries = {
        sym: df for sym, df in stock_data_with_indicators.items() if sym in prev_day_entries
    }

    # 5. Generate and write trades for all trade parameter sets and evaluate trades in these tabs
    try:
        logger.info("Generating and writing trades, updating opened trades in these tabs...")
        new_trades_tab_names = []

        for trade_params in generate_trade_param_sets():
            new_trades, new_trades_tab_name = generate_trades(
                stock_data_for_entries,
                **trade_params
            )

            if new_trades.empty:
                logger.info(f"No trades generated for tab '{new_trades_tab_name}'. Skipping.")
                continue

            gs_manager.update_trades(
                new_trades=new_trades,
                tab_name=new_trades_tab_name,
                eval_func=evaluate_trades,
                eval_data=stock_data_with_indicators
            )

            new_trades_tab_names.append(new_trades_tab_name)

            time.sleep(3)

        logger.info("All trades written and updated successfully.\n")
    except Exception as e:
        logger.error(f"Could not process trades due to error: {e}")

    # 6. Evaluate trades in other tabs
    try:
        additional_trade_tabs = list(set(trade_tabs) - set(new_trades_tab_names))
        logger.info(f"Updating trades in other {len(additional_trade_tabs)} tabs...")

        for trade_tab in additional_trade_tabs:
            gs_manager.update_trades(
                new_trades=pd.DataFrame(),
                tab_name=trade_tab,
                eval_func=evaluate_trades,
                eval_data=stock_data_with_indicators
            )

        logger.info("All trades updated successfully.\n")
    except Exception as e:
        logger.error(f"Could not update trades due to error: {e}")

    logger.info("Pipeline finished successfully.") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily stock trading pipeline.")
    parser.add_argument(
        "--filter_sets",
        nargs="+",
        default=DEFAULT_FILTER_SETS,
        required=False,
        help="List of filters to apply for potential entries"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="List of symbols to process (default test symbols)"
    )
    parser.add_argument(
        "--sheet_url",
        type=str,
        default="https://docs.google.com/spreadsheets/d/16HZfFIu37ZG7kRgLDI9SqS5xPhb3pn3iY2ubOymVseU/edit#gid=1171524967",
        help="Google Sheets URL"
    )

    args = parser.parse_args()

    result = main(
        sheet_url=args.sheet_url,
        filter_sets=args.filter_sets,
        symbols_list=args.symbols,
    )
