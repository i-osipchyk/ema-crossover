import os
import pandas as pd
from typing import List
import logging
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials

from .tools import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gs_client() -> gspread.Client:
    """
    Authenticate and return a Google Sheets client using service account credentials.

    Returns:
        gspread.Client: Authorized Google Sheets client.
    """
    creds_path = os.environ.get("GOOGLE_SHEETS_API_CREDENTIALS")
    if not creds_path:
        raise RuntimeError("GOOGLE_SHEETS_API_CREDENTIALS not set in environment!")
    
    creds = Credentials.from_service_account_file(
        creds_path,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    return gspread.authorize(creds)

def write_potential_entries(
    symbols: List[str], filter_set: List[str], sheet_url: str, sheet_tab: str = "Potential Setups"
) -> None:
    """
    Append today's potential entry symbols to Google Sheets as a single row.

    Args:
        symbols (List[str]): List of stock symbols.
        filter_set (List[str]): List of filters.
        sheet_url (str): Full URL of the Google Sheet.
        sheet_tab (str, optional): Tab name inside the sheet. Defaults to "Potential Setups".
    """
    if not symbols:
        logger.info("No symbols to write.")
        return

    client = get_gs_client()
    worksheet = client.open_by_url(sheet_url).worksheet(sheet_tab)

    today_str = datetime.now().strftime("%d-%m-%Y")
    row = [today_str, ", ".join(symbols), ", ".join(filter_set)]

    worksheet.append_row(row, value_input_option="USER_ENTERED")
    logger.info("✅ Wrote %s, %s to '%s' for %s.", ", ".join(symbols), ", ".join(filter_set), sheet_tab, today_str)

def get_previous_trading_day_entries(
    sheet_url: str, sheet_tab: str = "Potential Setups"
) -> List[str]:
    """
    Fetch symbols from the previous trading day row in Google Sheets.

    Args:
        sheet_url (str): Google Sheets URL.
        sheet_tab (str, optional): Tab name. Defaults to "Potential Setups".

    Returns:
        List[str]: List of symbols from the previous trading day.
    """
    client = get_gs_client()
    worksheet = client.open_by_url(sheet_url).worksheet(sheet_tab)
    rows = worksheet.get_all_records()  # list of dicts [{"Date": ..., "Symbols": ...}, ...]

    if not rows:
        return []

    # Parse dates safely
    for row in rows:
        try:
            row["Date"] = datetime.strptime(row["Date"], "%d-%m-%Y").date()
        except Exception:
            row["Date"] = None

    # Find the most recent previous trading day
    today = datetime.now().date()
    if today.weekday() == 0:   # Monday
        prev_day = today - timedelta(days=4)
    elif today.weekday() == 5: # Saturday
        prev_day = today - timedelta(days=2)
    elif today.weekday() == 6: # Sunday
        prev_day = today - timedelta(days=3)
    else:                      # Tue-Fri
        prev_day = today - timedelta(days=1)
    while prev_day.weekday() > 4:  # Skip weekends
        prev_day -= timedelta(days=1)

    prev_rows = [row for row in rows if row["Date"] == prev_day]
    if not prev_rows:
        logger.warning("No entries found for %s.", prev_day)
        return []

    # Merge symbols from all rows, remove duplicates
    all_symbols = set()
    for row in prev_rows:
        symbols = [s.strip() for s in row["Symbols"].split(",") if s.strip()]
        all_symbols.update(symbols)

    return sorted(all_symbols)

def write_trades_to_sheet(
    trades_df: pd.DataFrame,
    sheet_url: str,
    tab_name: str
) -> str:
    """
    Append trade DataFrame to Google Sheets. Creates a new tab if needed.
    
    Args:
        trades_df (pd.DataFrame): DataFrame of trades.
        sheet_url (str): Google Sheets URL.
    Returns:
        str: Name of the sheet tab where trades were written.
    """
    if trades_df.empty:
        print("No trades to write.")
        return ""
    
    # Authenticate using helper
    client = get_gs_client()

    # Open spreadsheet
    spreadsheet = client.open_by_url(sheet_url)

    # Tab name
    # tab_name = "Trades"

    # Create tab if it doesn't exist
    try:
        worksheet = spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=50)
        print(f"Created new tab: {tab_name}")

    df_to_write = trades_df.copy()

    # Check existing rows
    existing_rows = worksheet.get_all_values()
    existing_df = pd.DataFrame(existing_rows[1:], columns=existing_rows[0])
    
    # Filter out trades that already in the sheet, by Symbol and Date 
    if not existing_df.empty:
        existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_to_write['Date'] = pd.to_datetime(df_to_write['Date']).dt.strftime("%Y-%m-%d %H:%M:%S")

        mask = ~df_to_write.set_index(['Symbol', 'Date']).index.isin(
            existing_df.set_index(['Symbol', 'Date']).index
        )
        df_to_write = df_to_write[mask]
    
    if existing_df.empty:  # sheet is empty → write headers + data
        all_values = [df_to_write.columns.tolist()] + df_to_write.values.tolist()
        start_row = 1
    else:  # sheet already has data → append only data
        all_values = df_to_write.values.tolist()
        start_row = len(existing_rows) + 1

    # Write to sheet
    worksheet.update(
        f"A{start_row}",
        all_values,
        value_input_option="USER_ENTERED"
    )

    print(f"✅ Appended {len(df_to_write)} trades to tab '{tab_name}'")
    return tab_name

def read_trade_tabs(sheet_url: str, tab_name: str = "Trade Tabs List"):
    """
    Read all trade tabs from the 'Trade Tabs List' worksheet.
    Creates the worksheet with a header if it does not exist.
    """
    client = get_gs_client()
    spreadsheet = client.open_by_url(sheet_url)

    try:
        worksheet = spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        # Create the worksheet with header row
        worksheet = spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=1)
        worksheet.append_row(["Trade Tabs"], value_input_option="USER_ENTERED")
        logger.info("📄 Created new worksheet: %s", tab_name)

    # Get all values (first row = headers)
    all_values = worksheet.get_all_values()
    if not all_values or len(all_values) < 2:
        logger.warning("⚠️ No trade tabs found in '%s'.", tab_name)
        return []

    # Flatten values (one tab per row)
    trade_tabs = [row[0] for row in all_values[1:] if row]
    logger.info("📑 Found %s trade tabs", len(trade_tabs))

    return trade_tabs

def write_trade_tab(sheet_url: str, tab_to_write: str, list_tab_name: str = "Trade Tabs List"):
    """
    Append a trade tab name to the 'Trade Tabs List' sheet if it doesn't exist yet.
    Creates the sheet if missing.
    """
    client = get_gs_client()
    spreadsheet = client.open_by_url(sheet_url)

    try:
        worksheet = spreadsheet.worksheet(list_tab_name)
    except gspread.WorksheetNotFound:
        # Create the worksheet with header row
        worksheet = spreadsheet.add_worksheet(title=list_tab_name, rows=1000, cols=1)
        worksheet.append_row(["Trade Tabs"], value_input_option="USER_ENTERED")
        logger.info("📄 Created new worksheet: %s", list_tab_name)

    # Get all values (skip header row)
    all_values = worksheet.get_all_values()
    existing_tabs = {row[0] for row in all_values[1:] if row} if len(all_values) > 1 else set()

    if tab_to_write not in existing_tabs:
        worksheet.append_row([tab_to_write], value_input_option="USER_ENTERED")
        logger.info("✅ Added new trade tab: %s", tab_to_write)
    else:
        logger.info("ℹ️ Trade tab already exists: %s", tab_to_write)

def read_trades_from_sheet(sheet_url: str, tab_name: str = "Trades") -> pd.DataFrame:
    """
    Read trades from a Google Sheets tab into a pandas DataFrame.

    Args:
        sheet_url (str): Google Sheets URL.
        tab_name (str): Name of the worksheet tab to read. Defaults to "Trades".

    Returns:
        pd.DataFrame: DataFrame containing trades.
    """
    # Authenticate
    client = get_gs_client()

    # Open spreadsheet and worksheet
    spreadsheet = client.open_by_url(sheet_url)
    try:
        worksheet = spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        raise RuntimeError(f"Worksheet '{tab_name}' not found in spreadsheet!")

    # Get all values (first row = headers)
    all_values = worksheet.get_all_values()
    if not all_values or len(all_values) < 2:
        print("⚠️ No trade data found.")
        return pd.DataFrame()

    headers, data = all_values[0], all_values[1:]

    # Build DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Try convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")  # keep strings intact

    # Try convert date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df

def update_trades_sheet(sheet_url: str, evaluated_trades: pd.DataFrame, tab_name: str = "Trades"):
    """
    Update Google Sheet trades tab with evaluated trades.
    Only open trades (Position Left > 0) are updated; closed trades are untouched.

    Args:
        sheet_url (str): Google Sheets URL.
        evaluated_trades (pd.DataFrame): Trades after evaluation.
        tab_name (str): Worksheet tab name.
    """
    if 'Date' not in evaluated_trades.columns.to_list():
        evaluated_trades = evaluated_trades.reset_index()
        
    client = get_gs_client()
    spreadsheet = client.open_by_url(sheet_url)

    try:
        worksheet = spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        raise RuntimeError(f"Worksheet '{tab_name}' not found!")

    # Read all existing data
    all_values = worksheet.get_all_values()
    if not all_values or len(all_values) < 2:
        print("⚠️ No data in sheet to update.")
        return

    headers, data = all_values[0], all_values[1:]
    df_sheet = pd.DataFrame(data, columns=headers)

    # Convert numeric columns in sheet
    for col in df_sheet.columns:
        df_sheet[col] = pd.to_numeric(df_sheet[col], errors="ignore")

    if "Date" in df_sheet.columns:
        df_sheet["Date"] = pd.to_datetime(df_sheet["Date"], errors="coerce")

    # Merge updates: only open trades
    # Identify trades in sheet by Symbol + Date (or another unique identifier)
    key_cols = ["Date", "Symbol"]
    df_sheet.set_index(key_cols, inplace=True)
    
    evaluated_trades.set_index(key_cols, inplace=True)

    # Update only open trades (Position Left > 0)
    for idx, row in evaluated_trades.iterrows():
        # if row["Position Left"] > 0:
        if idx in df_sheet.index:
            df_sheet.loc[idx] = row  # overwrite the row
        else:
            df_sheet.loc[idx] = row  # new trade, append

    # Reset index
    df_sheet.reset_index(inplace=True)

    # Convert datetime columns to string
    for col in df_sheet.columns:
        if pd.api.types.is_datetime64_any_dtype(df_sheet[col]):
            df_sheet[col] = df_sheet[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Write back to sheet
    all_values_to_write = [df_sheet.columns.tolist()] + df_sheet.values.tolist()
    all_values_to_write = replace_nan_with_none(all_values_to_write)
    worksheet.update(f"A1", all_values_to_write, value_input_option="USER_ENTERED")

    print(f"✅ Sheet '{tab_name}' updated with evaluated trades (open trades updated, closed trades unchanged).")
