import numpy as np
import pandas as pd
from typing import List
from datetime import datetime
from itertools import product


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

def generate_trades(
    stock_data_for_entries, 
    stop_rule: str = 'crossover', 
    sl_offset_pc: float = 0.0, 
    max_tp: int = 5,
    skip_tp: int = 1,
    be_level_offset: float = 1.0,
    tp_coef: float = 2.0,
    position_size: float = 100.0,
    risk_usd: float = 5.0,
    position_sizing_method: str = 'risk'
) -> pd.DataFrame:
    """
    Generate trades from stock data with entry, SL, TP levels, TP sizes.

    Args:
        stock_data_for_entries (dict): Mapping symbol -> DataFrame with OHLC and indicators.
        stop_rule (str): Rule for stop loss placement ("crossover" or "previous").
        sl_offset_pc (float): Stop loss offset in % of risk.
        max_tp (int): Number of take profit levels.
        skip_tp (int): Number of TPs to skip at start.
        be_level_offset (float): Offset coefficient used with risk to define level when BE level is activated.
        tp_coef (float): Coefficient for geometric TP size distribution.
        position_size (float): Default position size.
        risk_usd (float): Risk in USD to determine position sizing.
        position_sizing_method (str): Method to define position sizing.
    Returns:
        pd.DataFrame: Trades with entry, stop loss, BE, risk, TPs and TP sizes.
    """
    today_trades = []

    for sym, df in stock_data_for_entries.items():
        if len(df) < 3:
            continue

        pre_cross_day = df.iloc[-3]
        cross_day = df.iloc[-2]
        today = df.iloc[-1]

        # Entry price
        if today["High"] >= cross_day["High"]:
            entry_price = max(cross_day["High"], today["Open"])
        else:
            continue

        # Stop loss
        if stop_rule == "crossover":
            stop_loss = cross_day["Low"]
        elif stop_rule == "previous":
            stop_loss = pre_cross_day["Low"]
        else:
            raise ValueError("Invalid stop rule")

        # Apply SL offset
        stop_loss -= sl_offset_pc / 100 * (entry_price - stop_loss)

        # Filter invalid trades
        if stop_loss >= entry_price:
            continue

        # Risk & share number
        risk_per_share = entry_price - stop_loss

        if position_sizing_method == 'risk':
            share_n = risk_usd / risk_per_share
            position_size = entry_price * share_n
        else:
            share_n = position_size / entry_price

        total_risk = risk_per_share * share_n

        # BE price and level
        be_price = entry_price
        be_level = entry_price + be_level_offset * risk_per_share

        # Save trade
        trade = {
            # "Date": datetime.now().strftime("%Y-%m-%d 00:00:00"),
            "Date": pd.to_datetime(df['Datetime'].values[-1]).strftime("%Y-%m-%d 00:00:00"),
            "Symbol": df["Symbol"].values[0], 
            "Entry Price": round(entry_price, 2),
            "Shares": round(share_n, 2),
            "Position Size": round(position_size, 2),
            "Stop Loss": round(stop_loss, 2),
            "Risk": round(total_risk, 2),
            "Risk per Share": round(risk_per_share, 2),
            "BE Price": round(be_price, 2),
            "Position Left": 1.0,
            "Realized": None,
            "Unrealized": None,
            "TP Reached": None,
            "BE Level": round(be_level, 2),
            "Stop Rule": stop_rule, 
            "Offset": sl_offset_pc, 
            "Max TP": max_tp,
            "Skip TP": skip_tp,
            "TP Coef": tp_coef
        }

        today_trades.append(trade)

    # Create DataFrame
    df_trades = pd.DataFrame(today_trades)

    # Reorder columns
    base_cols = [
        "Date", "Symbol", "Entry Price", "Shares", "Position Size", "Stop Loss", "Risk", "Risk per Share", "BE Price", "BE Level", 
        "Position Left", "Realized", "Unrealized", "TP Reached",
        "Stop Rule", "Offset", "Max TP", "Skip TP", "TP Coef"
    ]

    tab_name = f"{stop_rule}_SL{sl_offset_pc}_TP{max_tp}_Skip{skip_tp}_BE{be_level_offset}_Coef{tp_coef}_PosSiz{position_sizing_method}"
    
    if df_trades.empty:
        print("No trades today.")
    else:
        df_trades = df_trades[base_cols]
    
    return df_trades, tab_name

def evaluate_trades(trades_df: pd.DataFrame, stock_data: dict) -> pd.DataFrame:
    """
    Evaluate open trades against stock data, updating realized/unrealized returns
    and TP/SL/BE/EMA exit conditions.
    """
    df = trades_df.copy()

    if df.empty:
        print('No trades found')
        return df

    # Filter open trades
    open_trades = df[df["Position Left"] > 0]

    for idx, trade in open_trades.iterrows():
        symbol = trade["Symbol"]
        entry_date = pd.to_datetime(trade["Date"])
        entry_price = trade["Entry Price"]
        stop_loss = trade["Stop Loss"]
        risk = trade["Risk"]
        max_tp = int(trade["Max TP"])
        skip_tp = int(trade["Skip TP"])
        tp_coef = float(trade.get("TP Coef", 1.0))
        shares_n = trade["Shares"]
        pos_size = trade['Position Size']

        # print(f"\n🔎 Evaluating trade {idx} | {symbol} | Entry: {entry_price} | SL: {stop_loss} | Risk: {risk}")

        # Stock data for symbol
        if symbol not in stock_data:
            print(f"❌ No stock data for {symbol}, skipping...")
            continue
        df_sym = stock_data[symbol].copy()
        df_sym["Datetime"] = pd.to_datetime(df_sym["Datetime"])

        # Filter rows after entry
        df_sym = df_sym[df_sym["Datetime"] >= entry_date].reset_index(drop=True)

        # Build TP levels
        tp_levels = [entry_price + (i + 1 + skip_tp) * risk for i in range(max_tp)]
        tp_sizes = tp_distribution(max_tp, skip_tp, coef=tp_coef)

        # Track trade state
        position_left = 1.0
        realized = 0.0

        # Handle TP Reached history
        tp_reached_raw = trade.get("TP Reached", "")
        if pd.isna(tp_reached_raw) or tp_reached_raw == "":
            tp_reached = []
        elif isinstance(tp_reached_raw, str):
            # Split by comma if multiple exits are stored in one cell
            tp_reached = [x.strip() for x in tp_reached_raw.split(",") if x.strip()]
        else:
            # Already a single label (e.g., "SL", "BE") or unexpected type
            tp_reached = [str(tp_reached_raw)]

        be_price = entry_price

        for _, day in df_sym.iterrows():
            date, high, low, close = day["Datetime"], day["High"], day["Low"], day["Close"]

            # Stop Loss check
            if low <= stop_loss:
                realized += shares_n * position_left * stop_loss - entry_price * shares_n
                position_left = 0.0
                if stop_loss < entry_price:
                    tp_reached.append("SL")
                elif stop_loss > entry_price:
                    tp_reached.append("Trailing_SL")
                else:
                    tp_reached.append("BE")

                unrealized = 0.0
                # print(f"🛑 {date.date()} | SL hit at {stop_loss}, closing trade.")
                break

            # Take Profit checks
            for i, tp in enumerate(tp_levels):
                tp_label = f"TP{i+1}"
                if tp_label not in tp_reached and high >= tp:
                    exit_size = tp_sizes[i]
                    realized += shares_n * exit_size * tp
                    position_left -= exit_size
                    tp_reached.append(tp_label)
                    # print(f"✅ {date.date()} | {tp_label} hit at {tp}, exited {exit_size*100:.1f}% of position. Left: {position_left:.2f}")

                    # Update stop loss (trailing rule)
                    stop_loss = tp - 2 * risk if i >= 1 else be_price
                    # print(f"↔️ New Stop Loss set to {stop_loss}")

                    # If last TP reached, trade is closed
                    if i + 1 == max_tp:
                        position_left = 0.0
                        unrealized = 0.0
                        # print(f"🏁 Max TP{max_tp} reached, closing trade.")
                        break

            # EMA crossover check
            if day["EMA_Close_8"] < day["EMA_Close_20"]:
                realized += shares_n * position_left * close
                position_left = 0.0
                unrealized = 0.0
                tp_reached.append("EMA_Cross")
                # print(f"📉 {date.date()} | EMA crossover exit at {close}, closing trade.")
                break

            # unrealized = position_left * shares_n * close - pos_size
            unrealized = (close - entry_price) * shares_n

        # Update trade row
        df.at[idx, "Position Left"] = position_left
        df.at[idx, "Realized"] = realized
        df.at[idx, "Unrealized"] = unrealized # position_left * pos_size * df_sym.iloc[-1]["Close"]
        df.at[idx, "TP Reached"] = ",".join(tp_reached)

        # print(f"📊 Final state: Realized={realized:.2f}, Unrealized={df.at[idx,'Unrealized']:.2f}, Position Left={position_left:.2f}, TP Reached={df.at[idx,'TP Reached']}")
    return df

def generate_trade_param_sets():
    """
    Generate all trade parameter combinations.
    """
    sl_offsets = [-20, 0]
    max_tps = [4, 5]
    skip_tps = [1, 2]

    param_sets = []
    for sl, tp, skip in product(sl_offsets, max_tps, skip_tps):
        param_sets.append({
            "sl_offset_pc": float(sl),
            "max_tp": int(tp),
            "skip_tp": int(skip)
        })

    return param_sets
