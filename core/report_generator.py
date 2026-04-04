"""
HTML report generator for backtest results.

Reads the template from docs/report_template.html, maps NautilusTrader
backtest results to the template's expected JSON format, and produces
a self-contained HTML report file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def _parse_nautilus_value(value) -> float:
    """Parse a NautilusTrader Money/Quantity value to float.

    Handles strings like "150.00 USD", Decimal, int, float.
    """
    if value is None:
        return 0.0
    s = str(value).strip()
    # Strip currency suffix (e.g. "150.00 USD" -> "150.00")
    parts = s.split()
    try:
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def _format_timestamp(ts) -> str:
    """Convert a NautilusTrader timestamp to DD-MM-YYYY HH:MM:SS format."""
    if ts is None:
        return ""
    try:
        dt = pd.to_datetime(ts, utc=True)
        return dt.strftime("%d-%m-%Y %H:%M:%S")
    except Exception:
        return str(ts)


def _resolve_column(df: pd.DataFrame, candidates: list[str], default=None):
    """Find the first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return default


def _build_orderbook(all_results: dict) -> list[dict]:
    """Build the ORDERBOOK trade list from all strategy results.

    Each closed position becomes one trade record in the template's format.
    """
    trades = []

    for strategy_name, results in all_results.items():
        positions_report = results.get("positions_report")
        if positions_report is None or positions_report.empty:
            continue

        df = positions_report.reset_index()

        # Resolve column names (NautilusTrader may use different naming)
        col_instrument = _resolve_column(df, ["instrument_id", "InstrumentId", "instrument"])
        col_side = _resolve_column(df, ["entry", "side", "Side"])
        col_qty = _resolve_column(df, ["peak_qty", "quantity", "Quantity", "qty"])
        col_avg_open = _resolve_column(df, ["avg_px_open", "AvgPxOpen", "avg_open"])
        col_avg_close = _resolve_column(df, ["avg_px_close", "AvgPxClose", "avg_close"])
        col_pnl = _resolve_column(df, ["realized_pnl", "RealizedPnl", "realized_return", "pnl"])
        col_ts_open = _resolve_column(df, ["ts_opened", "TsOpened", "opened_time", "ts_init"])
        col_ts_close = _resolve_column(df, ["ts_closed", "TsClosed", "closed_time", "ts_last"])
        col_id = _resolve_column(df, ["id", "Id", "position_id", "index"])

        for idx, row in df.iterrows():
            raw_instrument = str(row[col_instrument]) if col_instrument else strategy_name
            # Clean up instrument id (e.g. "BTCUSD.CRYPTO" -> symbol="BTCUSD", exchange="CRYPTO")
            parts = raw_instrument.split(".")
            symbol = parts[0] if parts else strategy_name
            exchange = parts[1] if len(parts) > 1 else ""

            pnl = _parse_nautilus_value(row[col_pnl]) if col_pnl else 0.0
            qty = _parse_nautilus_value(row[col_qty]) if col_qty else 1.0
            entry_time = _format_timestamp(row[col_ts_open]) if col_ts_open else ""
            exit_time = _format_timestamp(row[col_ts_close]) if col_ts_close else ""

            trade = {
                "USERID": "UID001",
                "SYMBOL": symbol,
                "EXCHANGE": exchange,
                "TRANSACTION": str(row[col_side]) if col_side else "BUY",
                "LOTS": qty,
                "MULTIPLIER": 1.0,
                "QUANTITY": qty * 1.0,
                "OrderID": str(row[col_id]) if col_id else str(idx),
                "ENTRY TIME": entry_time,
                "ENTRY PRICE": _parse_nautilus_value(row[col_avg_open]) if col_avg_open else 0.0,
                "ENTRY REASON": "Strategy Signal",
                "OPTION TYPE": "",
                "STRIKE": "",
                "PORTFOLIO NAME": strategy_name,
                "STRATEGY": strategy_name,
                "EXIT TIME": exit_time,
                "AVG EXIT PRICE": _parse_nautilus_value(row[col_avg_close]) if col_avg_close else 0.0,
                "EXIT REASON": "Strategy Signal",
                "PNL": pnl,
                "_IS_HEDGE": False,
                "_PARENT_ID": "",
            }
            trades.append(trade)

    # Sort by entry time
    trades.sort(key=lambda t: t["ENTRY TIME"])
    return trades


def build_orderbook_dataframe(all_results: dict) -> pd.DataFrame:
    """Build an order book DataFrame matching the full CSV schema.

    Parameters
    ----------
    all_results : dict
        Strategy name -> results dict (from run_backtest).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching the order book CSV spec.
    """
    trades = _build_orderbook(all_results)
    if not trades:
        return pd.DataFrame()

    column_order = [
        "USERID", "SYMBOL", "EXCHANGE", "TRANSACTION", "QUANTITY", "LOTS",
        "MULTIPLIER", "OrderID", "ENTRY TIME", "ENTRY PRICE", "ENTRY REASON",
        "OPTION TYPE", "STRIKE", "PORTFOLIO NAME", "STRATEGY",
        "EXIT TIME", "AVG EXIT PRICE", "EXIT REASON", "PNL",
        "_IS_HEDGE", "_PARENT_ID",
    ]
    return pd.DataFrame(trades).reindex(columns=column_order, fill_value="")


def build_logs_dataframe(
    all_results: dict,
    run_timestamp: str = "",
    user_id: str = "UID001",
) -> pd.DataFrame:
    """Build a trading logs DataFrame from fills and positions reports.

    Generates ENTRY and EXIT log rows from the backtest results, matching
    the log CSV schema: Timestamp, Backtest_Timestamp, Log Type, Message,
    UserID, Strategy Tag, Option Portfolio, Strike.

    Parameters
    ----------
    all_results : dict
        Strategy name -> results dict (from run_backtest).
    run_timestamp : str
        The wall-clock time when the backtest was executed (for the Timestamp column).
    user_id : str
        Default user ID for the logs.

    Returns
    -------
    pd.DataFrame
        DataFrame with log columns.
    """
    from datetime import datetime as _dt

    if not run_timestamp:
        run_timestamp = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    logs: list[dict] = []

    for strategy_name, results in all_results.items():
        fills = results.get("fills_report")
        if fills is None or fills.empty:
            continue

        df = fills.reset_index()

        col_instrument = _resolve_column(df, ["instrument_id", "InstrumentId", "instrument"])
        col_side = _resolve_column(df, ["side", "Side"])
        col_qty = _resolve_column(df, ["filled_qty", "quantity", "Quantity", "qty"])
        col_price = _resolve_column(df, ["avg_px", "AvgPx", "price"])
        col_ts = _resolve_column(df, ["ts_last", "ts_init", "TsLast"])
        col_reduce = _resolve_column(df, ["is_reduce_only", "IsReduceOnly"])
        col_order_id = _resolve_column(df, ["venue_order_id", "VenueOrderId", "order_id"])
        col_position_id = _resolve_column(df, ["position_id", "PositionId"])

        for _, row in df.iterrows():
            raw_instrument = str(row[col_instrument]) if col_instrument else ""
            parts = raw_instrument.split(".")
            symbol = parts[0] if parts else ""

            bt_time = _format_timestamp(row[col_ts]) if col_ts else ""
            side = str(row[col_side]) if col_side else ""
            price = _parse_nautilus_value(row[col_price]) if col_price else 0.0
            qty = _parse_nautilus_value(row[col_qty]) if col_qty else 0.0
            is_reduce = bool(row[col_reduce]) if col_reduce else False
            order_id = str(row[col_order_id]) if col_order_id else ""
            position_id = str(row[col_position_id]) if col_position_id else ""

            if is_reduce:
                log_type = "TRADING"
                action = "EXIT"
                msg = (
                    f"{action} | {side} {qty} {symbol} @ {price:.2f}"
                    f" | OrderID={order_id} | PositionID={position_id}"
                )
            else:
                log_type = "TRADING"
                action = "ENTRY"
                msg = (
                    f"{action} | {side} {qty} {symbol} @ {price:.2f}"
                    f" | OrderID={order_id} | PositionID={position_id}"
                )

            logs.append({
                "Timestamp": run_timestamp,
                "Backtest_Timestamp": bt_time,
                "Log Type": log_type,
                "Message": msg,
                "UserID": user_id,
                "Strategy Tag": strategy_name,
                "Option Portfolio": strategy_name,
                "Strike": "",
            })

    if not logs:
        return pd.DataFrame()

    result = pd.DataFrame(logs)
    result.sort_values("Backtest_Timestamp", inplace=True, ignore_index=True)
    return result


def _build_summary(orderbook: list[dict]) -> dict:
    """Build per-date summary from the orderbook trades.

    Returns a dict keyed by DD-MM-YYYY date strings with max/min PNL info.
    """
    if not orderbook:
        return {}

    # Group trades by date
    date_groups: dict[str, list[dict]] = {}
    for trade in orderbook:
        date_str = trade["ENTRY TIME"].split(" ")[0] if trade["ENTRY TIME"] else ""
        if not date_str:
            continue
        date_groups.setdefault(date_str, []).append(trade)

    summary = {}
    for date_str, day_trades in date_groups.items():
        pnl_values = [float(t["PNL"]) for t in day_trades]
        cumulative = []
        running = 0.0
        for pnl in pnl_values:
            running += pnl
            cumulative.append(running)

        max_cum = max(cumulative) if cumulative else 0.0
        min_cum = min(cumulative) if cumulative else 0.0
        max_idx = cumulative.index(max_cum)
        min_idx = cumulative.index(min_cum)

        # Build per-portfolio leg stats for Min/Max PNL column
        portfolio_stats: dict = {}
        for trade in day_trades:
            pf = trade.get("PORTFOLIO NAME", "")
            if pf not in portfolio_stats:
                portfolio_stats[pf] = {"max_pnl": 0.0, "leg_stats": {}}
            pnl = float(trade["PNL"])
            order_id = trade.get("OrderID", "")
            entry_time = trade.get("ENTRY TIME", "")
            if order_id:
                portfolio_stats[pf]["leg_stats"][order_id] = {
                    "min_pnl": pnl,
                    "max_pnl": pnl,
                    "min_pnl_time": entry_time,
                    "max_pnl_time": entry_time,
                }
            pf_pnls = [float(t["PNL"]) for t in day_trades if t.get("PORTFOLIO NAME") == pf]
            portfolio_stats[pf]["max_pnl"] = sum(pf_pnls)

        summary[date_str] = {
            "max_pnl": max_cum,
            "min_pnl": min_cum,
            "max_pnl_time": day_trades[max_idx]["ENTRY TIME"],
            "min_pnl_time": day_trades[min_idx]["ENTRY TIME"],
            "portfolio_stats": portfolio_stats,
        }

    return summary


def generate_report(
    all_results: dict,
    backtest_name: str = "Backtest",
    template_path: str | None = None,
) -> str:
    """Generate a self-contained HTML backtest report.

    Parameters
    ----------
    all_results : dict
        Strategy name -> results dict (from run_backtest).
    backtest_name : str
        Title for the report.
    template_path : str, optional
        Path to the HTML template. Defaults to docs/report_template.html.

    Returns
    -------
    str
        Complete HTML string ready to be saved or served.
    """
    if template_path is None:
        template_path = str(Path(__file__).resolve().parent.parent / "docs" / "report_template.html")

    template = Path(template_path).read_text(encoding="utf-8")

    orderbook = _build_orderbook(all_results)
    summary = _build_summary(orderbook)
    logs: list = []

    html = template.replace("{{ ORDERBOOK_JSON }}", json.dumps(orderbook, default=str))
    html = html.replace("{{ SUMMARY_JSON }}", json.dumps(summary, default=str))
    html = html.replace("{{ LOGS_JSON }}", json.dumps(logs, default=str))
    html = html.replace("{{BACKTEST_NAME}}", backtest_name)

    return html
