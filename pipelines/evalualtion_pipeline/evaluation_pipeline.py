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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_FILTER_SETS = [
    ["Above_EMA_34", "Above_EMA_50", "Above_EMA_200", "MACD_Positive", "Volume", "Price", "ATR"],
    ["Above_EMA_34", "Above_EMA_50", "Above_EMA_200", "MACD_Positive", "MACD_Signal_Negative", "Volume", "Price", "ATR"],
]


# TODO: handle Google Sheets API requests limit

def main(
    filter_sets: List[List[str]],
    symbols_list: List[str] = None,
    sheet_url: str = "https://docs.google.com/spreadsheets/d/16HZfFIu37ZG7kRgLDI9SqS5xPhb3pn3iY2ubOymVseU/edit#gid=1171524967",
    potential_tab: str = "Potential Setups"
) -> Dict[str, Any]:
    """
    Full daily pipeline:
    - Download and label stock data
    - Find potential entries
    - Write today's potential entries to Google Sheets
    - Fetch previous trading day's entries
    - Generate trades for previous day entries
    - Append trades to Google Sheets

    Args:
        filters (List[str]): List of filter sets to apply for potential entries.
        symbols_list (List[str], optional): List of symbols. Defaults to a test list.
        sheet_url (str): Google Sheets URL.
        potential_tab (str): Tab name for potential entries.

    Returns:
        Dict[str, Any]: {"today": list of today's potential entries,
                         "previous_day": list of previous trading day entries,
                         "trades_df": DataFrame of trades}
    """
    logger.info("⬇️ Downloading fresh data...")
    shift = 0

    symbols_list = read_list_from_file("data/all-symbols-june-2025.txt")
    # symbols_list = ['COR', 'HIG', 'NJR', 'AROC', 'CB', 'DUK', 'CF', 'KGS', 'JNJ', 'TTE', 'NOC', 'VTR', 'DVN', 'SQM', 'BKH', 'ORLY', 'NXPI', 'ROST', 'GH']

    # 1. Download and label all data
    stock_data = download_data(
        symbols=symbols_list,
        period="500d",
        interval="1d",
        batch_size=100,
    )

    stock_data_labeled = apply_to_dict(stock_data, process_symbol_df)

    # 2. Find potential entries based on different sets of filters
    potential_entries_with_filters = []

    for filter_set in filter_sets:
        potential_entries, filter_set = find_potential_entries(
            stock_data_labeled,
            shift=shift,
            filter_set=filter_set,
        )

        if potential_entries:
            potential_entries_with_filters.append((potential_entries, filter_set))

        # logger.info("📈 Potential entries for filter set %s: %s", filter_set, potential_entries)

    # 3. Write potential entries and their filters
    for potential_entries, filter_set in potential_entries_with_filters:
        write_potential_entries(
            symbols=potential_entries,
            filter_set=filter_set,
            sheet_url=sheet_url,
            sheet_tab=potential_tab
        )

        time.sleep(1)

    # 4. Read list of all trade tabs
    trade_tabs = read_trade_tabs(sheet_url=sheet_url)

    # 5. For each trade tab make evaluation and update it
    for trade_tab in trade_tabs:
        # Get existing trades, evaluate them and update sheet
        existing_trades = read_trades_from_sheet(sheet_url=sheet_url, tab_name=trade_tab)
        existing_trades_evaluated = evaluate_trades(existing_trades, stock_data_labeled)
        update_trades_sheet(sheet_url=sheet_url, tab_name=trade_tab, evaluated_trades=existing_trades_evaluated)

        time.sleep(1)

    # 6. Get all potential entries from previous day
    prev_day_entries = get_previous_trading_day_entries(
        sheet_url=sheet_url,
        sheet_tab=potential_tab
    )

    logger.info("📅 Previous trading day entries: %s", prev_day_entries)

    # Filter stock data for entries 
    stock_data_for_entries = {sym: df for sym, df in stock_data_labeled.items() if sym in prev_day_entries}

    # 7. Generate trades for all trade tabs
    for trade_params in generate_trade_param_sets():
        trades_df, generated_trades_tab = generate_trades(
            stock_data_for_entries,
            **trade_params
        )

        if not trades_df.empty:
            write_trades_to_sheet(
                trades_df=trades_df,
                sheet_url=sheet_url,
                tab_name=generated_trades_tab
            )

        time.sleep(1)

        write_trade_tab(sheet_url=sheet_url, tab_to_write=generated_trades_tab)

        time.sleep(1)

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
    parser.add_argument(
        "--potential_tab",
        type=str,
        default="Potential Setups",
        help="Tab name for potential entries"
    )

    args = parser.parse_args()

    result = main(
        filter_sets=args.filter_sets,
        symbols_list=args.symbols,
        sheet_url=args.sheet_url,
        potential_tab=args.potential_tab
    )

    logger.info("Pipeline finished successfully.")
