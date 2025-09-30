import sys
import math
from datetime import datetime, timedelta, timezone


def get_tqdm():
    """Return tqdm that works for notebook or terminal."""
    try:
        # If running inside Jupyter/IPython kernel
        if "ipykernel" in sys.modules:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
    except ImportError:
        # fallback in case notebook extra is not installed
        from tqdm import tqdm
    return tqdm

def replace_nan_with_none(data):
    """
    Replace NaN or Inf values with None in a list of lists.
    
    Args:
        data (list[list]): Nested list of values.
    Returns:
        list[list]: Same structure but with NaN/Inf replaced by None.
    """
    cleaned = []
    for row in data:
        new_row = []
        for v in row:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                new_row.append(None)
            else:
                new_row.append(v)
        cleaned.append(new_row)
    return cleaned

def get_last_trading_day(now: datetime | None = None) -> datetime:
    """
    Determine the last trading day based on current UTC time.

    Rules:
    - If run on Saturday or Sunday → go back to Thursday.
    - If run Mon–Fri before 13:30 UTC → go back 2 days.
    - Else → go back 1 day.
    - If the result falls on Sunday → go back 2 more days.
    - If the result falls on Saturday → go back 1 more day.

    Args:
        now (datetime, optional): Current time (default = now in UTC).

    Returns:
        datetime: Last trading day (date only, UTC).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    weekday = now.weekday()  # Monday=0 ... Sunday=6

    # Weekend handling
    if weekday == 5:  # Saturday
        target = now - timedelta(days=2)  # Thursday
    elif weekday == 6:  # Sunday
        target = now - timedelta(days=3)  # Thursday
    else:
        # Weekday logic
        if now.hour < 13 or (now.hour == 13 and now.minute < 30):
            target = now - timedelta(days=2)
        else:
            target = now - timedelta(days=1)

    # If we land on weekend, shift again
    if target.weekday() == 6:  # Sunday
        target -= timedelta(days=2)
    elif target.weekday() == 5:  # Saturday
        target -= timedelta(days=1)

    return target.date()
