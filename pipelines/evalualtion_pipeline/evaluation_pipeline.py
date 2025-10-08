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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    try:
        gs_manager = GoogleSheetsManager(sheet_url=sheet_url)
        logger.info("Successfully connected to Google Sheets.")
    except Exception as e:
        logger.error(f"Could not connect to Google Sheets due to error: {e}")

    logger.info("Downloading and labeling data...")

    # 1. Download and label all data
    try:
        symbols_list = read_list_from_file("data/all-symbols-june-2025.txt")

        stock_data = download_data(
            symbols=symbols_list,
            period="500d",
            interval="1d",
            batch_size=100,
        )

        stock_data_labeled = apply_to_dict(stock_data, add_indicators)
    except Exception as e:
        logger.error(f"Could not download and label data due to error: {e}") 

    # 2. Find potential entries based on different sets of filters
    potential_entries_with_filters = [
        (entries, fset)
        for fset in filter_sets
        if (entries := find_potential_entries(stock_data_labeled, filter_set=fset)[0])
    ]

    # 3. Write potential entries and their filters
    latest_day = next(iter(stock_data_labeled.values()))['Datetime'].values[-1] # Get latest day

    gs_manager.write_potential_entries(
        latest_day=latest_day,
        potential_entries_with_filters=potential_entries_with_filters
    )

    # 4. Read all potential entries from the previous trading day and all existing trade tabs
    prev_day_entries = gs_manager.read_previous_trading_day_entries()
    logger.info("Previous trading day entries: %s", prev_day_entries)

    trade_tabs = gs_manager.read_trade_tabs()
    logger.info(f"Found {len(trade_tabs)} trade tabs.")

    # TODO: Select data for generation in trade generation function
    # Filter stock data for these entries
    stock_data_for_entries = {
        sym: df for sym, df in stock_data_labeled.items() if sym in prev_day_entries
    }

    # 5. Generate and write trades for all trade parameter sets and evaluate trades in these tabs
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
            eval_data=stock_data_labeled
        )

        new_trades_tab_names.append(new_trades_tab_name)

        time.sleep(3)

    # 6. Evaluate trades in other tabs
    additional_trade_tabs = list(set(trade_tabs) - set(new_trades_tab_names))
    logger.info(f"Found {len(additional_trade_tabs)} additional trade tabs. Evaluating trades...")

    for trade_tab in additional_trade_tabs:
        gs_manager.update_trades(
            new_trades=pd.DataFrame(),
            tab_name=trade_tab,
            eval_func=evaluate_trades,
            eval_data=stock_data_labeled
        )

    logger.info("All trades evaluated and updated successfully.")

    return 

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

    logger.info("Pipeline finished successfully.")
