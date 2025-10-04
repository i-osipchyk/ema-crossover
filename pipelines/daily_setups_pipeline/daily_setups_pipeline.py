import os
import glob
import argparse
from datetime import datetime
import pandas as pd
import warnings

from modules.download import *
from modules.indicators import *
from modules.pipeline import *

warnings.simplefilter(action="ignore", category=FutureWarning)


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

def main(args):
    filters = args.filters.split(",") if args.filters else DEFAULT_FILTERS
    shift = args.shift

    os.makedirs("daily_labeled", exist_ok=True)

    latest_file = get_latest_labeled_file()

    if market_data_is_fresh(latest_file):
        print("✅ Market not renewed yet, using latest labeled data...")
        stock_data_labeled = pd.read_pickle(latest_file)
    else:
        print("⬇️ Downloading fresh data...")

        symbols_list = read_list_from_file("data/all-symbols-june-2025.txt")

        stock_data = download_data(
            symbols=symbols_list,
            period="500d",
            interval="1d",
            batch_size=100,
        )

        stock_data_labeled = apply_to_dict(stock_data, add_indicators)

        # save snapshot with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = f"daily_labeled/labeled_{timestamp}.pkl"
        pd.to_pickle(stock_data_labeled, out_path)
        print(f"💾 Saved labeled data to {out_path}")

    # Run potential entries
    potential_entries = find_potential_entries(
        stock_data_labeled,
        shift=shift,
        filters=filters,
    )

    print("\n📈 Potential entries:")
    print(potential_entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily stock pipeline with filters")
    parser.add_argument("--filters", type=str, help="Comma-separated filters to apply")
    parser.add_argument("--shift", type=int, default=0, help="Shift for historical testing")
    args = parser.parse_args()
    main(args)
