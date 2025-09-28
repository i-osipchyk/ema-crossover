from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StopRule:
    CROSSOVER = "crossover"
    PREVIOUS = "previous"


def bullish_bearish_pairs(
    df: pd.DataFrame,
    short_ma_params: Tuple[int, str, str],
    long_ma_params: Tuple[int, str, str],
) -> List[Tuple[int, int]]:
    """
    Identify bullish-bearish crossover pairs from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with crossover signals.
        short_ma_params (tuple): Parameters for short moving average (period, source, method).
        long_ma_params (tuple): Parameters for long moving average (period, source, method).

    Returns:
        List[Tuple[int, int]]: List of (bullish_index, bearish_index) pairs.
    """
    short_p, short_s, short_m = short_ma_params
    long_p, long_s, long_m = long_ma_params

    short_ma_name = f"{short_m}_{short_s}_{short_p}"
    long_ma_name = f"{long_m}_{long_s}_{long_p}"
    signal_col = f"{short_ma_name}_{long_ma_name}_Crossover"

    if signal_col not in df.columns:
        logger.warning("Signal column %s not found in DataFrame", signal_col)
        return []

    bullish_indices = df.index[df[signal_col] == "Bullish"].tolist()
    bearish_indices = df.index[df[signal_col] == "Bearish"].tolist()

    if not bullish_indices:
        return []

    # Remove first Bearish if it comes before first Bullish
    if bearish_indices and bearish_indices[0] < bullish_indices[0]:
        bearish_indices.pop(0)

    # Add last index if no Bearish exists after last Bullish
    if not bearish_indices or bearish_indices[-1] < bullish_indices[-1]:
        bearish_indices.append(df.index[-1])

    # Ensure valid pairs
    pairs = [(b, r) for b, r in zip(bullish_indices, bearish_indices) if b < r]
    return pairs


def tp_distribution(max_tp: int, skip_tp: int, coef: float) -> List[float]:
    """
    Generate a geometric distribution of take-profit (TP) sizes.

    Args:
        max_tp (int): Total number of TP levels.
        skip_tp (int): Number of initial TP levels to skip (assigned 0 weight).
        coef (float): Geometric coefficient for distribution.

    Returns:
        List[float]: Normalized TP size weights for all levels.
    """
    if max_tp <= skip_tp:
        raise ValueError("max_tp must be greater than skip_tp")

    weights = np.array([coef**i for i in range(max_tp - skip_tp)])
    weights /= weights.sum()
    weights = [0.0] * skip_tp + weights.tolist()
    return weights


def simulate_trades(
    df: pd.DataFrame,
    short_ma_params: Tuple[int, str, str],
    long_ma_params: Tuple[int, str, str],
    stop_rule: str,
    sl_offset_pc: float,
    skip_tp: int,
    max_tp: int,
    tp_coef: float,
) -> List[Dict[str, Any]]:
    """
    Simulate trades based on moving average crossovers and trade rules.

    Args:
        df (pd.DataFrame): Price data with crossover and candle columns.
        short_ma_params (tuple): Parameters for short MA.
        long_ma_params (tuple): Parameters for long MA.
        stop_rule (str): Stop loss rule ("crossover" or "previous").
        sl_offset_pc (float): Stop loss offset percentage.
        skip_tp (int): Number of initial TP levels to skip.
        max_tp (int): Maximum number of take profit levels.
        tp_coef (float): Geometric coefficient for TP sizing.

    Returns:
        List[Dict[str, Any]]: List of simulated trade dictionaries.
    """
    trades: List[Dict[str, Any]] = []
    pairs = bullish_bearish_pairs(df, short_ma_params, long_ma_params)

    tp_sizes = tp_distribution(max_tp=max_tp, skip_tp=skip_tp, coef=tp_coef)

    for bull_idx, bear_idx in pairs:
        symbol = df.get("Symbol", pd.Series(["Unknown"])).iloc[0]
        trade: Dict[str, Any] = {
            "Symbol": symbol,
            "StopRule": stop_rule,
            "Offset": sl_offset_pc,
            "SkipTP": skip_tp,
            "MaxTP": max_tp,
            "TPCoef": tp_coef,
            "ExitTypes": [],
            "Exits": [],
        }

        try:
            pre_cross_day = df.iloc[bull_idx - 1]
            bull_cross_day = df.iloc[bull_idx]
            entry_day = df.iloc[bull_idx + 1] if bull_idx + 1 < len(df) else None
        except IndexError:
            logger.debug("Skipping trade: index out of range")
            continue

        if entry_day is None:
            continue

        # Entry condition
        entry_price = (
            max(bull_cross_day["High"], entry_day["Open"])
            if entry_day["High"] > bull_cross_day["High"]
            else None
        )
        if entry_price is None:
            continue

        # Stop loss
        if stop_rule == StopRule.CROSSOVER:
            stop_loss = bull_cross_day["Low"]
        elif stop_rule == StopRule.PREVIOUS:
            stop_loss = pre_cross_day["Low"]
        else:
            raise ValueError(f"Invalid stop rule: {stop_rule}")

        stop_loss -= sl_offset_pc / 100 * (entry_price - stop_loss)

        if stop_loss >= entry_price:
            continue  # invalid risk-reward

        risk = entry_price - stop_loss
        tp_levels = [entry_price + (i + 1) * risk for i in range(max_tp)]
        be_price = entry_price

        # Populate trade details
        trade.update(
            {
                "Datetime": entry_day["Datetime"],
                "EntryPrice": entry_price,
                "StopLoss": stop_loss,
                "Risk": risk,
                "MaxTP": tp_levels[-1],
                "CrossoverVolume": bull_cross_day["Volume"],
                "CrossoverColor": bull_cross_day["Candle_Color"],
            }
        )

        # Track remaining position size
        remaining_size = 1.0

        # Simulate exits
        for j in range(bull_idx + 1, bear_idx + 1):
            day = df.iloc[j]

            # Stop loss hit (exit remaining position)
            if day["Low"] <= stop_loss:
                exit_type = (
                    "Trailing_SL" if stop_loss > entry_price else
                    "SL" if stop_loss < entry_price else
                    "BE"
                )
                trade["ExitTypes"].append(exit_type)
                trade["Exits"].append(
                    {"Type": exit_type, "Date": day["Datetime"], "Price": stop_loss, "Size": remaining_size, "Realized": stop_loss * remaining_size / entry_price}
                )
                break

            # Take profits
            for idx, (tp, size) in enumerate(zip(tp_levels, tp_sizes)):
                if size == 0.0:
                    continue  # skip TP levels
                if day["High"] >= tp and f"TP{idx+1}" not in trade["ExitTypes"]:
                    trade["ExitTypes"].append(f"TP{idx+1}")
                    trade["Exits"].append(
                        {"Type": f"TP{idx+1}", "Date": day["Datetime"], "Price": tp, "Size": size, "Realized": tp * size / entry_price}
                    )
                    remaining_size -= size
                    stop_loss = tp - 2 * risk if idx >= 1 else be_price
                    if idx + 1 == max_tp:  # all TPs reached
                        remaining_size = 0.0
                        break

            # EMA Cross exit (close remaining position)
            if j == bear_idx and remaining_size > 0:
                trade["ExitTypes"].append("EMA_Cross")
                trade["Exits"].append(
                    {"Type": "EMA_Cross", "Date": day["Datetime"], "Price": day["Close"], "Size": remaining_size, "Realized": stop_loss * remaining_size / entry_price}
                )
                break

        trades.append(trade)

    return trades


def flatten_trades(all_results):
    flattened = []

    for symbol, trades in all_results.items():
        for trade in trades:  # each trade is a dict
            for exit_info in trade["Exits"]:
                row = {
                    "Symbol": trade["Symbol"],
                    "StopRule": trade["StopRule"],
                    "Offset": trade["Offset"],
                    "SkipTP": trade["SkipTP"],
                    "MaxTP": trade["MaxTP"],
                    "TPCoef": trade["TPCoef"],
                    "Datetime": trade["Datetime"],
                    "EntryPrice": trade["EntryPrice"],
                    "Risk": trade["Risk"],
                    "StopLoss": trade["StopLoss"],
                    "CrossoverVolume": trade.get("CrossoverVolume", None),
                    "CrossoverColor": trade.get("CrossoverColor", None),
                    "ExitType": exit_info["Type"],
                    "ExitDate": exit_info["Date"],
                    "ExitPrice": exit_info["Price"],
                    "ExitSize": exit_info["Size"],
                    "Realized": exit_info["Realized"]
                }
                flattened.append(row)

    return pd.DataFrame(flattened)


def aggregate_trades(exits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trade executions into one row per trade (grouped by Symbol and Datetime).

    Computes weighted exit price, return %, R-multiple, last exit date, and concatenated
    exit types. If any exit price is missing, return metrics are set to NaN.

    Args:
        exits_df (pd.DataFrame): Dataframe with a single entry exit pair in a row.
    Returns:
        pd.DataFrame: One row per trade with aggregated metrics.
    """
    grouped = exits_df.groupby(["Symbol", "Datetime"], group_keys=False)

    results = []
    for _, group in grouped:
        if group["ExitPrice"].isna().any():
            results.append({
                "Symbol": group["Symbol"].iloc[0],
                "EntryDate": group["Datetime"].iloc[0],
                "EntryPrice": group["EntryPrice"].iloc[0],
                "StopLoss": group["StopLoss"].iloc[0],
                "Risk": group["Risk"].iloc[0],
                "WeightedExitPrice": np.nan,
                "Return%": np.nan,
                "R_Multiple": np.nan,
                "ExitDate": pd.NaT,
                "ExitTypes": None
            })
            continue

        total_position = group["ExitSize"].sum()
        if total_position == 0:
            total_position = 1.0  # avoid div/0

        weighted_exit_price = (group["ExitPrice"] * group["ExitSize"]).sum() / total_position

        entry_price = group["EntryPrice"].iloc[0]
        risk = group["Risk"].iloc[0]

        results.append({
            "Symbol": group["Symbol"].iloc[0],
            "EntryDate": group["Datetime"].iloc[0],
            "EntryPrice": entry_price,
            "StopLoss": group["StopLoss"].iloc[0],
            "Risk": risk,
            "WeightedExitPrice": weighted_exit_price,
            "Return%": (weighted_exit_price - entry_price) / entry_price * 100,
            "R_Multiple": (weighted_exit_price - entry_price) / risk if risk != 0 else np.nan,
            "ExitDate": group["ExitDate"].max(),
            "ExitTypes": ",".join(group["ExitType"].tolist())
        })

    return pd.DataFrame(results)


def add_indicators_to_trades(trades_df: pd.DataFrame, 
                             historical_data_dict: dict) -> pd.DataFrame:
    """
    Enrich trades with indicator values from historical data.

    For each trade in `trades_df`, this function looks up the corresponding
    historical data (by symbol and entry date) in `historical_data_dict`
    and merges available indicator columns into the trade row.

    Args:
        trades_df (pd.DataFrame): DataFrame of trades. Must include
            "Symbol" and "EntryDate" columns.
        historical_data_dict (dict[str, pd.DataFrame]): Dictionary mapping
            symbols to their historical DataFrames. Each DataFrame must
            contain a "Datetime" column and indicator columns.

    Returns:
        pd.DataFrame: New DataFrame of trades with historical indicators
        (if found) merged into each row. Trades without matching
        historical data remain unchanged.
    """
    enriched_trades = []

    for _, trade in trades_df.iterrows():
        sym = trade["Symbol"]
        dt = trade["EntryDate"]

        # start with trade row
        row = trade.to_dict()

        # lookup historical dataframe for this symbol
        if sym in historical_data_dict:
            hist_df = historical_data_dict[sym]
            match = hist_df.loc[hist_df["Datetime"] == dt]

            if not match.empty:
                # take the first row of matching historical data
                hist_data = match.iloc[0].to_dict()

                # merge into trade row
                row.update(hist_data)

        enriched_trades.append(row)

    return pd.DataFrame(enriched_trades)


def apply_trade_filters(trades_df: pd.DataFrame, 
                        filter_set: List[str], 
                        min_crossover_vol: float = 500_000, 
                        min_price: float = 20, 
                        max_atr: float = 5.77) -> pd.DataFrame:
    """
    Filter a trades DataFrame using a set of conditions.

    Args:
        trades_df (pd.DataFrame): Trade data with indicator columns 
            (e.g., "Above_EMA_Close_34", "MACD", "Volume", "ATR%").
        filter_set (set): Filters to apply. Available:
            {"Above_EMA_34", "Above_EMA_50", "Above_EMA_200",
             "MACD_Positive", "MACD_Signal_Negative", "Volume", "Price", "ATR"}.
        min_crossover_vol (float, optional): Minimum volume threshold if "Volume" filter is set.
        min_price (float, optional): Minimum close price if "Price" filter is set.
        max_atr (float, optional): Maximum ATR% if "ATR" filter is set.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows that pass all filters.
    """
    df = trades_df.copy()

    if "Above_EMA_34" in filter_set:
        df = df[df['Above_EMA_Close_34'] == True]
    if "Above_EMA_50" in filter_set:
        df = df[df['Above_EMA_Close_34'] == True]
    if "Above_EMA_200" in filter_set:
        df = df[df['Above_EMA_Close_34'] == True]
    if "MACD_Positive" in filter_set:
        df = df[df['MACD'] >= 0]
    if "MACD_Signal_Negative" in filter_set:
        df = df[df['MACD_Signal'] <= 0]
    if "Volume" in filter_set:
        df = df[df['Volume'] >= min_crossover_vol]
    if "Price" in filter_set:
        df = df[df['EntryPrice'] >= min_price]
    if "ATR" in filter_set:
        df = df[df['ATR%'] < max_atr]

    return df
