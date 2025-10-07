import os
import pandas as pd
import numpy as np
from typing import List
import logging
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials

from .tools import *


class GoogleSheetsManager:

    def __init__(self, sheet_url: str, test_tab: str = "Test",
                 potential_setups_tab: str = "Potential Setups",
                 trade_tabs_list_tab: str = "Trade Tabs List"):
        self.sheet_url = sheet_url
        self.test_tab = test_tab
        self.potential_setups_tab = potential_setups_tab
        self.trade_tabs_list_tab = trade_tabs_list_tab
        self.client = self._get_gs_client()
        self.logger = logging.getLogger()


    def _get_gs_client(self) -> gspread.Client:
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


    def write_potential_entries(self, latest_day: np.datetime64,
        potential_entries_with_filters: List[tuple[List[str], List[str]]]) -> None:
        """
        Append multiple potential entry sets (symbols + filters) to Google Sheets in one batch.

        Args:
            potential_entries_with_filters (List[Tuple[List[str], List[str]]]): List of (symbols, filters) tuples.
            latest_day (np.datetime64): Date of setups.
        """
        if not potential_entries_with_filters:
            self.logger.info("No potential entries to write.")
            return

        worksheet = self.client.open_by_url(self.sheet_url).worksheet(self.potential_setups_tab)

        # Convert date to desired string format
        latest_day_str = pd.to_datetime(latest_day).to_pydatetime().strftime("%d-%m-%Y")

        # Prepare all rows to append
        rows_to_append = []
        for symbols, filters in potential_entries_with_filters:
            if not symbols:
                continue
            row = [latest_day_str, ", ".join(symbols), ", ".join(filters)]
            rows_to_append.append(row)

        if not rows_to_append:
            self.logger.info("No valid symbol lists to write.")
            return

        # Write all rows in one request
        worksheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")

        self.logger.info(f"✅ Wrote {len(rows_to_append)} potential setups to '{self.potential_setups_tab}' for {latest_day_str}.")


    def read_previous_trading_day_entries(self) -> List[str]:
        """
        Fetch symbols from the previous trading day.

        Returns:
            List[str]: List of symbols from the previous trading day.
        """
        worksheet = self.client.open_by_url(self.sheet_url).worksheet(self.potential_setups_tab)
        rows = worksheet.get_all_records()

        if not rows:
            return []

        # Parse dates safely
        for row in rows:
            try:
                row["Date"] = datetime.strptime(row["Date"], "%d-%m-%Y").date()
            except Exception:
                row["Date"] = None

        # Find the most recent previous trading day
        prev_day = get_last_trading_day(datetime.now())

        prev_rows = [row for row in rows if row["Date"] == prev_day]
        if not prev_rows:
            self.logger.warning("No entries found for %s.", prev_day)
            return []

        # Merge symbols from all rows, remove duplicates
        all_symbols = set()
        for row in prev_rows:
            symbols = [s.strip() for s in row["Symbols"].split(",") if s.strip()]
            all_symbols.update(symbols)
        
        self.logger.info(f"Read {len(all_symbols)} symbols for {prev_day}.")

        return sorted(all_symbols)


    def read_trades(self, tab_name: str) -> pd.DataFrame:
        """
        Read trades from a tab into a pandas DataFrame.

        Args:
            tab_name (str): Name of the worksheet tab to read.

        Returns:
            pd.DataFrame: DataFrame containing trades.
        """
        # Open spreadsheet and worksheet
        spreadsheet = self.client.open_by_url(self.sheet_url)
        try:
            worksheet = spreadsheet.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            raise gspread.WorksheetNotFound(f"Worksheet '{tab_name}' not found in spreadsheet!")

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
            df[col] = pd.to_numeric(df[col], errors="ignore")

        # Try convert date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        return df
    

    def _write_df_to_sheet(self, spreadsheet, tab_name: str, df: pd.DataFrame) -> None:
        """Overwrite a Google Sheets tab with a DataFrame."""
        worksheet = spreadsheet.worksheet(tab_name)
        all_values = [df.columns.tolist()] + df.astype(str).fillna("").values.tolist()
        worksheet.update("A1", all_values, value_input_option="USER_ENTERED")


    def update_trades(self, new_trades: pd.DataFrame, tab_name: str, 
                      eval_func: callable, eval_data: dict) -> None:
        """
        Check current state of the tab.
            * If does not exist, create and add new trades.
            * If exists but empty, add new trades.
            * If exists and has data, evaluate open trades, add new ones and remove duplicates.

        Args:
            new_trades (pd.DataFrame): New trades to add.
            tab_name (str): Name of the Google Sheet tab.
            eval_func (callable): Function to evaluate existing trades.
            eval_data (dict): Data used for trade evaluation.
        """
        spreadsheet = self.client.open_by_url(self.sheet_url)

        try:
            existing_df = self.read_trades(tab_name)
            if existing_df.empty:
                self.logger.info(f"🆕 '{tab_name}' found but empty → writing new trades.")
                self._write_df_to_sheet(spreadsheet, tab_name, new_trades)
                return
            
            existing_df_eval = eval_func(existing_df, eval_data)
            combined_df = pd.concat([existing_df_eval, new_trades], ignore_index=True)

            if {"Date", "Symbol"}.issubset(combined_df.columns):
                combined_df = combined_df.drop_duplicates(subset=["Date", "Symbol"], keep="first")

            if "Date" in combined_df.columns:
                combined_df = combined_df.sort_values(by="Date")
            
            self._write_df_to_sheet(spreadsheet, tab_name, combined_df)

            self.logger.info(f"✅ '{tab_name}' updated successfully with evaluated + new trades.")

        except gspread.WorksheetNotFound:
            self.logger.info(f"📄 '{tab_name}' not found → creating and writing new trades.")
            spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=50)
            self._write_df_to_sheet(spreadsheet, tab_name, new_trades)
            self.logger.info(f"✅ Created new tab '{tab_name}' and wrote {len(new_trades)} new trades.")
            