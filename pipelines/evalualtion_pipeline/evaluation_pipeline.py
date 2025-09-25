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

from modules.download import *
from modules.indicators import *
from modules.pipeline import *
from modules.tools import *
from modules.simulation import *
from modules.google_sheet_tools import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_FILTERS = [
    "Above_EMA_34",
    "Above_EMA_50",
    "Above_EMA_200",
    "MACD_Positive",
    # "MACD_Signal_Negative",
    "Volume",
    "Price",
    "ATR",
]


def main(
    filters: List[str],
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
        filters (List[str]): List of filters to apply for potential entries.
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
    # symbols_list = ['WELL', 'MET', 'ETR', 'NVDA', 'QRVO', 'WSBC', 'TARS', 'COR', 'FERG', 'HIG', 'TXNM', 'NJR', 'AROC']

    # Download OHLCV 
    stock_data = download_data(
        symbols=symbols_list,
        period="500d",
        interval="1d",
        batch_size=100,
    )

    # Label data 
    stock_data_labeled = apply_to_dict(stock_data, process_symbol_df)

    # Run potential entries 
    potential_entries = find_potential_entries(
        stock_data_labeled,
        shift=shift,
        filters=filters,
    )

    logger.info("📈 Potential entries: %s", potential_entries)

    # Save potential entries to Google Sheets 
    if potential_entries:
        write_potential_entries(
            symbols=potential_entries,
            sheet_url=sheet_url,
            sheet_tab=potential_tab
        )

    # Get existing trades, evaluate them and update sheet
    existing_trades = read_trades_from_sheet(sheet_url)
    existing_trades_evaluated = evaluate_trades(existing_trades, stock_data_labeled)
    update_trades_sheet(sheet_url, existing_trades_evaluated)

    #  Get previous trading day's entries 
    prev_day_entries = get_previous_trading_day_entries(
        sheet_url=sheet_url,
        sheet_tab=potential_tab
    )

    logger.info("📅 Previous trading day entries: %s", prev_day_entries)

    # Filter stock data for entries 
    stock_data_for_entries = {sym: df for sym, df in stock_data_labeled.items() if sym in prev_day_entries}

    # Generate trades DataFrame 
    trades_df = generate_trades(
        stock_data_for_entries,
    )

    # Write trades to Google Sheets
    if not trades_df.empty:
        write_trades_to_sheet(
            trades_df=trades_df,
            sheet_url=sheet_url
        )

    return {
        "today": potential_entries,
        "previous_day": prev_day_entries,
        "trades_df": trades_df
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily stock trading pipeline.")
    parser.add_argument(
        "--filters",
        nargs="+",
        default=DEFAULT_FILTERS,
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
        filters=args.filters,
        symbols_list=args.symbols,
        sheet_url=args.sheet_url,
        potential_tab=args.potential_tab
    )

    logger.info("Pipeline finished successfully.")
