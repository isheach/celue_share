from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# -----------------------------
# Configuration / data classes
# -----------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the stock-level sector rotation backtest")
    p.add_argument("--config", default="config.yaml", help="YAML config path")
    return p.parse_args()


@dataclass
class PositionLot:
    lot_id: str
    code: str
    name: str
    sector: str
    shares: float
    entry_date: pd.Timestamp
    entry_price: float
    signal_date: pd.Timestamp
    scheduled_exit_date: pd.Timestamp
    last_price: float
    overdue: bool = False
    freeze_start_date: pd.Timestamp | None = None
    freeze_event_index: int | None = None


@dataclass
class PendingOrder:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    scheduled_exit_date: pd.Timestamp
    selected_sectors: list[dict[str, Any]]


@dataclass
class Sleeve:
    sleeve_id: int
    cash: float
    positions: list[PositionLot] = field(default_factory=list)
    pending: PendingOrder | None = None


# -----------------------------
# Helpers
# -----------------------------


def normalize_ts(x: Any) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def chosen_st_column(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("st_detection", "flag_or_name")).lower()
    if mode == "flag":
        return "is_st_flag"
    if mode == "name":
        return "is_st_name"
    if mode == "flag_or_name":
        return "is_st"
    raise ValueError(f"Unknown st_detection: {mode}")


def weighting_column(mode: str, lagged: bool) -> str | None:
    mode = mode.lower()
    if mode == "equal":
        return None
    prefix = "prev_" if lagged else ""
    mapping = {
        "free_float_mcap": f"{prefix}free_float_mcap",
        "float_mcap": f"{prefix}float_mcap",
        "total_mcap": f"{prefix}total_mcap",
    }
    if mode not in mapping:
        raise ValueError(f"Unknown weighting mode: {mode}")
    return mapping[mode]


def apply_industry_aliases(series: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    aliases = cfg.get("industry_aliases") or {}
    if not aliases:
        return series
    return series.replace(aliases)


def positive_num(v: Any) -> bool:
    return pd.notna(v) and float(v) > 0.0


def signal_eligible_mask(day_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    mask = (
        (~day_df["is_suspended"].fillna(True).astype(bool))
        & day_df["industry"].notna()
        & (day_df["close"].fillna(0) > 0)
        & (day_df["pre_close"].fillna(0) > 0)
    )
    if bool(cfg.get("require_positive_volume", True)):
        mask &= day_df["volume"].fillna(0) > 0
    if bool(cfg.get("exclude_st", False)):
        mask &= ~day_df[chosen_st_column(cfg)].fillna(False).astype(bool)
    return mask


def buyable_row(row: pd.Series, cfg: dict[str, Any]) -> bool:
    if bool(row.get("is_suspended", True)):
        return False
    if not positive_num(row.get("open")):
        return False
    if bool(cfg.get("require_positive_volume", True)) and not positive_num(row.get("volume")):
        return False
    if bool(cfg.get("exclude_st", False)) and bool(row.get(chosen_st_column(cfg), False)):
        return False
    return True


def sellable_row(row: pd.Series, price_col: str, cfg: dict[str, Any]) -> bool:
    # ST is NEVER used to block an exit.  ST filtering only controls signal/buy eligibility.
    if bool(row.get("is_suspended", True)):
        return False
    if not positive_num(row.get(price_col)):
        return False
    if bool(cfg.get("require_positive_volume", True)) and not positive_num(row.get("volume")):
        return False
    return True


def compute_member_weights(
    members: pd.DataFrame,
    mode: str,
) -> dict[str, float]:
    """
    Every eligible stock must participate.

    For cap weighting, if even one eligible stock lacks a positive cap value,
    the WHOLE sector falls back to equal weighting.  This avoids silently
    dropping a stock and preserves the user's 'all tradable stocks' rule.
    """
    if members.empty:
        return {}

    mode = mode.lower()
    if mode == "equal":
        w = 1.0 / len(members)
        return {str(code): w for code in members["code"]}

    col = weighting_column(mode, lagged=False)
    assert col is not None
    vals = pd.to_numeric(members[col], errors="coerce")
    valid = vals.notna() & (vals > 0)

    if valid.sum() != len(members):
        w = 1.0 / len(members)
        return {str(code): w for code in members["code"]}

    total = float(vals.sum())
    if not math.isfinite(total) or total <= 0:
        w = 1.0 / len(members)
        return {str(code): w for code in members["code"]}

    return {
        str(code): float(value) / total
        for code, value in zip(members["code"], vals, strict=False)
    }


def round_shares(raw_shares: float, cfg: dict[str, Any]) -> float:
    if bool(cfg.get("allow_fractional_shares", True)):
        return max(0.0, raw_shares)
    lot_size = int(cfg.get("lot_size", 100))
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    lots = math.floor(raw_shares / lot_size)
    return float(max(0, lots * lot_size))


# -----------------------------
# Signal construction
# -----------------------------


def build_sector_daily(
    con: duckdb.DuckDBPyConnection,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, list[tuple[str, float]]]]:
    st_col = chosen_st_column(cfg)
    exclude_st = bool(cfg.get("exclude_st", False))
    require_volume = bool(cfg.get("require_positive_volume", True))
    weight_mode = str(cfg.get("sector_return_weighting", "free_float_mcap")).lower()
    weight_col = weighting_column(weight_mode, lagged=True)

    clauses = [
        "industry IS NOT NULL",
        "NOT is_suspended",
        "stock_return IS NOT NULL",
    ]
    if require_volume:
        clauses.append("volume > 0")
    if exclude_st:
        clauses.append(f"NOT {st_col}")
    where_sql = " AND ".join(clauses)

    if weight_col is None:
        query = f"""
        SELECT
            date,
            industry,
            COUNT(*) AS n_constituents,
            AVG(stock_return) AS equal_return,
            AVG(stock_return) AS sector_return,
            FALSE AS cap_weight_fallback
        FROM daily
        WHERE {where_sql}
        GROUP BY date, industry
        ORDER BY date, industry
        """
    else:
        # Cap weighting is used only when EVERY eligible stock in that sector/day
        # has a positive lagged cap. Otherwise the whole sector/day falls back to
        # equal weighting so no eligible stock disappears from the calculation.
        query = f"""
        WITH grouped AS (
            SELECT
                date,
                industry,
                COUNT(*) AS n_constituents,
                AVG(stock_return) AS equal_return,
                SUM(CASE WHEN {weight_col} > 0 THEN 1 ELSE 0 END) AS valid_weight_count,
                SUM(CASE WHEN {weight_col} > 0 THEN {weight_col} ELSE 0 END) AS weight_sum,
                SUM(CASE WHEN {weight_col} > 0 THEN stock_return * {weight_col} ELSE 0 END) AS weighted_ret_sum
            FROM daily
            WHERE {where_sql}
            GROUP BY date, industry
        )
        SELECT
            date,
            industry,
            n_constituents,
            equal_return,
            CASE
                WHEN valid_weight_count = n_constituents AND weight_sum > 0
                THEN weighted_ret_sum / weight_sum
                ELSE equal_return
            END AS sector_return,
            CASE
                WHEN valid_weight_count = n_constituents AND weight_sum > 0
                THEN FALSE ELSE TRUE
            END AS cap_weight_fallback
        FROM grouped
        ORDER BY date, industry
        """

    sector_daily = con.execute(query).fetchdf()
    sector_daily["date"] = pd.to_datetime(sector_daily["date"]).dt.normalize()
    sector_daily["industry"] = apply_industry_aliases(sector_daily["industry"], cfg)

    # If the user explicitly aliases two historical labels into one label, merge
    # the already-built sector rows using constituent-count weighting.  Default
    # config has no aliases, so the raw historical labels are preserved.
    if (cfg.get("industry_aliases") or {}) and sector_daily.duplicated(["date", "industry"]).any():
        sector_daily["weighted_component"] = (
            sector_daily["sector_return"] * sector_daily["n_constituents"]
        )
        sector_daily = (
            sector_daily.groupby(["date", "industry"], as_index=False)
            .agg(
                n_constituents=("n_constituents", "sum"),
                equal_return=("equal_return", "mean"),
                weighted_component=("weighted_component", "sum"),
                cap_weight_fallback=("cap_weight_fallback", "max"),
            )
        )
        sector_daily["sector_return"] = (
            sector_daily["weighted_component"] / sector_daily["n_constituents"]
        )
        sector_daily.drop(columns=["weighted_component"], inplace=True)

    trading_days = con.execute("SELECT date FROM trading_days ORDER BY date").fetchdf()
    trading_days["date"] = pd.to_datetime(trading_days["date"]).dt.normalize()
    all_days = pd.DatetimeIndex(trading_days["date"])

    ret_pivot = sector_daily.pivot(index="date", columns="industry", values="sector_return")
    ret_pivot = ret_pivot.reindex(all_days)

    n_pivot = sector_daily.pivot(index="date", columns="industry", values="n_constituents")
    n_pivot = n_pivot.reindex(index=all_days, columns=ret_pivot.columns)

    window = int(cfg.get("momentum_days", 10))
    if window <= 0:
        raise ValueError("momentum_days must be positive")

    # Exactly the last N GLOBAL trading days. A sector with a missing day does
    # not get a momentum value for that window.
    momentum = (1.0 + ret_pivot).rolling(window=window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1.0

    min_constituents = int(cfg.get("min_sector_constituents", 1))
    top_n = int(cfg.get("top_n_sectors", 3))
    signals_by_date: dict[pd.Timestamp, list[tuple[str, float]]] = {}

    for dt in all_days:
        row = momentum.loc[dt].dropna()
        if row.empty:
            signals_by_date[normalize_ts(dt)] = []
            continue
        counts = n_pivot.loc[dt]
        eligible_names = counts[counts >= min_constituents].index
        row = row[row.index.isin(eligible_names)]
        top = row.sort_values(ascending=False).head(top_n)
        signals_by_date[normalize_ts(dt)] = [(str(k), float(v)) for k, v in top.items()]

    mom_long = (
        momentum.stack(dropna=False, future_stack=False)
        .rename("momentum")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "industry"})
    )
    sector_daily = sector_daily.merge(mom_long, on=["date", "industry"], how="left")

    return sector_daily, pd.DataFrame({"date": all_days}), signals_by_date


# -----------------------------
# Event-driven execution engine
# -----------------------------


def fetch_day(con: duckdb.DuckDBPyConnection, dt: pd.Timestamp, cfg: dict[str, Any]) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT
            code, name, industry, date,
            open, high, low, close, volume, amount, pre_close,
            limit_up, limit_down,
            total_mcap, float_mcap, free_float_mcap,
            is_suspended, is_st_flag, is_st_name, is_st
        FROM daily
        WHERE date = ?
        ORDER BY code
        """,
        [dt.date()],
    ).fetchdf()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["industry"] = apply_industry_aliases(df["industry"], cfg)
    return df


def sell_lot(
    sleeve: Sleeve,
    lot: PositionLot,
    price: float,
    dt: pd.Timestamp,
    execution: str,
    cfg: dict[str, Any],
    trades: list[dict[str, Any]],
) -> None:
    gross = lot.shares * price
    sell_fee = gross * float(cfg.get("sell_fee_rate", 0.000086))
    sell_tax = gross * float(cfg.get("sell_tax_rate", 0.0))
    net = gross - sell_fee - sell_tax
    sleeve.cash += net
    trades.append(
        {
            "date": dt,
            "sleeve": sleeve.sleeve_id,
            "side": "SELL",
            "execution": execution,
            "code": lot.code,
            "name": lot.name,
            "sector": lot.sector,
            "shares": lot.shares,
            "price": price,
            "gross_notional": gross,
            "fee": sell_fee,
            "tax": sell_tax,
            "net_cash": net,
            "signal_date": lot.signal_date,
            "entry_date": lot.entry_date,
            "scheduled_exit_date": lot.scheduled_exit_date,
        }
    )
    sleeve.positions.remove(lot)


def execute_overdue_open_exits(
    sleeve: Sleeve,
    day_df: pd.DataFrame,
    dt: pd.Timestamp,
    cfg: dict[str, Any],
    trades: list[dict[str, Any]],
    suspension_events: list[dict[str, Any]],
    date_to_index: dict[pd.Timestamp, int],
) -> None:
    if day_df.empty:
        return
    rows = day_df.set_index("code", drop=False)
    for lot in list(sleeve.positions):
        if not lot.overdue:
            continue
        if lot.code not in rows.index:
            continue
        row = rows.loc[lot.code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if not sellable_row(row, "open", cfg):
            continue

        px = float(row["open"])
        sell_lot(sleeve, lot, px, dt, "OPEN_AFTER_SUSPENSION", cfg, trades)

        if lot.freeze_event_index is not None:
            ev = suspension_events[lot.freeze_event_index]
            ev["status"] = "resolved"
            ev["actual_exit_date"] = dt
            ev["actual_exit_price"] = px
            ev["delay_trading_days"] = (
                date_to_index[dt] - date_to_index[lot.scheduled_exit_date]
            )


def execute_pending_entry(
    sleeve: Sleeve,
    day_df: pd.DataFrame,
    dt: pd.Timestamp,
    cfg: dict[str, Any],
    trades: list[dict[str, Any]],
    lot_counter: list[int],
) -> None:
    pending = sleeve.pending
    if pending is None or pending.entry_date != dt:
        return

    if day_df.empty:
        # There should be a market row for every trading day. If not, retain cash
        # and discard this one-day entry instruction rather than using future data.
        sleeve.pending = None
        return

    rows = day_df.set_index("code", drop=False)
    starting_cash = sleeve.cash
    top_n = int(cfg.get("top_n_sectors", 3))
    buy_fee_rate = float(cfg.get("buy_fee_rate", 0.000086))

    for sector_order in pending.selected_sectors:
        sector = sector_order["sector"]
        signal_weights: dict[str, float] = sector_order["member_weights"]

        buyable_codes: list[str] = []
        for code in signal_weights:
            if code not in rows.index:
                continue
            row = rows.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if buyable_row(row, cfg):
                buyable_codes.append(code)

        if not buyable_codes:
            # This sector's 1/TopN cash slice remains cash.
            continue

        # Re-normalize the SIGNAL-DAY weights only across names actually buyable
        # at the next day's open.  We do not add stocks that were absent from the
        # signal-day universe.
        w_sum = sum(signal_weights[c] for c in buyable_codes)
        if w_sum <= 0:
            continue

        sector_budget = starting_cash / top_n
        for code in buyable_codes:
            row = rows.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            px = float(row["open"])
            norm_w = signal_weights[code] / w_sum
            budget = sector_budget * norm_w

            # budget is total cash spent including the configured buy fee.
            raw_shares = budget / (px * (1.0 + buy_fee_rate))
            shares = round_shares(raw_shares, cfg)
            if shares <= 0:
                continue

            gross = shares * px
            fee = gross * buy_fee_rate
            cash_cost = gross + fee
            if cash_cost > sleeve.cash + 1e-8:
                # Can happen only from integer-lot rounding or floating precision.
                affordable = sleeve.cash / (px * (1.0 + buy_fee_rate))
                shares = round_shares(affordable, cfg)
                if shares <= 0:
                    continue
                gross = shares * px
                fee = gross * buy_fee_rate
                cash_cost = gross + fee

            sleeve.cash -= cash_cost
            lot_counter[0] += 1
            lot = PositionLot(
                lot_id=f"S{sleeve.sleeve_id}-{lot_counter[0]}",
                code=str(code),
                name=str(row.get("name") or ""),
                sector=sector,
                shares=shares,
                entry_date=dt,
                entry_price=px,
                signal_date=pending.signal_date,
                scheduled_exit_date=pending.scheduled_exit_date,
                last_price=px,
            )
            sleeve.positions.append(lot)
            trades.append(
                {
                    "date": dt,
                    "sleeve": sleeve.sleeve_id,
                    "side": "BUY",
                    "execution": "OPEN",
                    "code": lot.code,
                    "name": lot.name,
                    "sector": sector,
                    "shares": shares,
                    "price": px,
                    "gross_notional": gross,
                    "fee": fee,
                    "tax": 0.0,
                    "net_cash": -cash_cost,
                    "signal_date": pending.signal_date,
                    "entry_date": dt,
                    "scheduled_exit_date": pending.scheduled_exit_date,
                }
            )

    sleeve.pending = None


def execute_scheduled_close_exits(
    sleeve: Sleeve,
    day_df: pd.DataFrame,
    dt: pd.Timestamp,
    cfg: dict[str, Any],
    trades: list[dict[str, Any]],
    suspension_events: list[dict[str, Any]],
) -> None:
    rows = day_df.set_index("code", drop=False) if not day_df.empty else None

    for lot in list(sleeve.positions):
        if lot.overdue or lot.scheduled_exit_date != dt:
            continue

        row = None
        if rows is not None and lot.code in rows.index:
            row = rows.loc[lot.code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

        if row is not None and sellable_row(row, "close", cfg):
            sell_lot(
                sleeve,
                lot,
                float(row["close"]),
                dt,
                "SCHEDULED_CLOSE",
                cfg,
                trades,
            )
            continue

        # The strategy wants to exit now but cannot.  Keep the real position.
        lot.overdue = True
        lot.freeze_start_date = dt
        lot.freeze_event_index = len(suspension_events)
        suspension_events.append(
            {
                "sleeve": sleeve.sleeve_id,
                "code": lot.code,
                "name": lot.name,
                "sector": lot.sector,
                "entry_date": lot.entry_date,
                "scheduled_exit_date": lot.scheduled_exit_date,
                "freeze_start_date": dt,
                "status": "frozen",
                "actual_exit_date": None,
                "actual_exit_price": None,
                "delay_trading_days": None,
            }
        )


def refresh_last_prices(sleeve: Sleeve, day_df: pd.DataFrame) -> None:
    if day_df.empty or not sleeve.positions:
        return
    rows = day_df.set_index("code", drop=False)
    for lot in sleeve.positions:
        if lot.code not in rows.index:
            continue
        row = rows.loc[lot.code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # Suspended rows often carry a stale close. That is acceptable for mark-to-
        # market; if no positive close exists we explicitly carry the last price.
        if positive_num(row.get("close")):
            lot.last_price = float(row["close"])


def sleeve_equity(sleeve: Sleeve) -> float:
    pos_value = sum(p.shares * p.last_price for p in sleeve.positions)
    return float(sleeve.cash + pos_value)


def build_pending_order(
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    selected: list[tuple[str, float]],
    day_df: pd.DataFrame,
    cfg: dict[str, Any],
    signal_log: list[dict[str, Any]],
    sleeve_id: int,
) -> PendingOrder | None:
    if not selected or day_df.empty:
        return None

    eligible = day_df.loc[signal_eligible_mask(day_df, cfg)].copy()
    if eligible.empty:
        return None

    constituent_mode = str(cfg.get("constituent_weighting", "free_float_mcap"))
    sector_orders: list[dict[str, Any]] = []

    for rank, (sector, momentum) in enumerate(selected, start=1):
        members = eligible.loc[eligible["industry"] == sector].copy()
        weights = compute_member_weights(members, constituent_mode)
        signal_log.append(
            {
                "signal_date": signal_date,
                "sleeve": sleeve_id,
                "rank": rank,
                "sector": sector,
                "momentum": momentum,
                "signal_day_members": len(members),
                "entry_date": entry_date,
                "scheduled_exit_date": exit_date,
            }
        )
        if not weights:
            continue
        sector_orders.append(
            {
                "sector": sector,
                "momentum": momentum,
                "member_weights": weights,
            }
        )

    if not sector_orders:
        return None
    return PendingOrder(signal_date, entry_date, exit_date, sector_orders)


# -----------------------------
# Metrics / output
# -----------------------------


def calc_metrics(equity: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if equity.empty:
        raise ValueError("No equity observations were produced")

    nav = equity["equity"] / initial_capital
    daily_ret = equity["equity"].pct_change().fillna(0.0)
    n = max(1, len(equity) - 1)
    final_equity = float(equity["equity"].iloc[-1])
    final_nav = final_equity / initial_capital
    ann = final_nav ** (252.0 / n) - 1.0 if final_nav > 0 else -1.0
    rolling_max = nav.cummax()
    dd = nav / rolling_max - 1.0
    max_dd = float(dd.min())
    vol = float(daily_ret.std(ddof=1))
    sharpe = float(daily_ret.mean() / vol * np.sqrt(252.0)) if vol > 0 else None

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "final_nav": final_nav,
        "total_return": final_nav - 1.0,
        "annualized_return": float(ann),
        "max_drawdown": max_dd,
        "sharpe_0rf": sharpe,
        "equity_observations": int(len(equity)),
    }


def json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, date)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    db_path = Path(cfg.get("cache_db", "./cache/market.duckdb"))
    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB cache not found: {db_path}. Run prepare_data.py first."
        )

    output_root = Path(cfg.get("output_dir", "./outputs"))
    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = output_root / run_id
    out.mkdir(parents=True, exist_ok=False)

    initial_capital = float(cfg.get("initial_capital", 1_000_000.0))
    hold_days = int(cfg.get("hold_days", 2))
    n_sleeves = int(cfg.get("staggered_portfolios", 2))
    if hold_days <= 0 or n_sleeves <= 0:
        raise ValueError("hold_days and staggered_portfolios must be positive")
    if n_sleeves != hold_days:
        print(
            "WARNING: staggered_portfolios != hold_days. This is allowed, but it no longer "
            "matches the video's standard staggered construction."
        )

    con = duckdb.connect(str(db_path), read_only=True)
    print("Building stock-derived sector returns and momentum signals ...")
    sector_daily, calendar_df, signals_by_date = build_sector_daily(con, cfg)

    all_days = [normalize_ts(x) for x in calendar_df["date"]]
    if not all_days:
        raise ValueError("No trading days in cache")
    date_to_index = {dt: i for i, dt in enumerate(all_days)}

    start = normalize_ts(cfg.get("signal_start_date", all_days[0]))
    configured_end = cfg.get("end_date")
    end = normalize_ts(configured_end) if configured_end else all_days[-1]

    run_days = [d for d in all_days if start <= d <= end]
    if not run_days:
        raise ValueError(f"No trading days between {start.date()} and {end.date()}")

    first_global_idx = date_to_index[run_days[0]]
    last_global_idx = date_to_index[run_days[-1]]

    sleeves = [
        Sleeve(i, initial_capital / n_sleeves)
        for i in range(n_sleeves)
    ]

    trades: list[dict[str, Any]] = []
    suspension_events: list[dict[str, Any]] = []
    signal_log: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    lot_counter = [0]

    print(
        f"Running event loop: {run_days[0].date()} -> {run_days[-1].date()} "
        f"({len(run_days)} trading days)"
    )

    for local_i, dt in enumerate(run_days):
        global_i = date_to_index[dt]
        day_df = fetch_day(con, dt, cfg)

        # 1) Earliest possible liquidation of positions that should have exited
        # on an earlier day but were frozen by suspension/no trading.
        for sleeve in sleeves:
            execute_overdue_open_exits(
                sleeve,
                day_df,
                dt,
                cfg,
                trades,
                suspension_events,
                date_to_index,
            )

        # 2) Execute yesterday's signal at today's open.
        for sleeve in sleeves:
            execute_pending_entry(
                sleeve,
                day_df,
                dt,
                cfg,
                trades,
                lot_counter,
            )

        # 3) Scheduled T+2 (for hold_days=2) exits happen at today's close.
        for sleeve in sleeves:
            execute_scheduled_close_exits(
                sleeve,
                day_df,
                dt,
                cfg,
                trades,
                suspension_events,
            )

        # 4) Mark whatever remains in the account at today's close.
        for sleeve in sleeves:
            refresh_last_prices(sleeve, day_df)

        # 5) Generate today's close signal only when a future entry AND scheduled
        # exit both exist inside the chosen backtest range.
        relative_signal_idx = local_i
        sleeve_id = relative_signal_idx % n_sleeves
        entry_global_i = global_i + 1
        exit_global_i = global_i + hold_days
        if entry_global_i <= last_global_idx and exit_global_i <= last_global_idx:
            selected = signals_by_date.get(dt, [])
            pending = build_pending_order(
                signal_date=dt,
                entry_date=all_days[entry_global_i],
                exit_date=all_days[exit_global_i],
                selected=selected,
                day_df=day_df,
                cfg=cfg,
                signal_log=signal_log,
                sleeve_id=sleeve_id,
            )
            if pending is not None:
                if sleeves[sleeve_id].pending is not None:
                    raise RuntimeError(
                        f"Sleeve {sleeve_id} already has a pending order on {dt.date()}"
                    )
                sleeves[sleeve_id].pending = pending

        sleeve_values = [sleeve_equity(s) for s in sleeves]
        row = {
            "date": dt,
            "equity": float(sum(sleeve_values)),
            "cash": float(sum(s.cash for s in sleeves)),
            "open_positions": int(sum(len(s.positions) for s in sleeves)),
            "frozen_positions": int(
                sum(sum(1 for p in s.positions if p.overdue) for s in sleeves)
            ),
        }
        for i, value in enumerate(sleeve_values):
            row[f"sleeve_{i}_equity"] = value
        equity_rows.append(row)

        if local_i % 250 == 0 or local_i == len(run_days) - 1:
            print(
                f"  {dt.date()}  equity={row['equity']:.2f}  "
                f"positions={row['open_positions']}  frozen={row['frozen_positions']}"
            )

    con.close()

    equity = pd.DataFrame(equity_rows)
    equity["nav"] = equity["equity"] / initial_capital
    equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
    equity["drawdown"] = equity["nav"] / equity["nav"].cummax() - 1.0

    trades_df = pd.DataFrame(trades)
    susp_df = pd.DataFrame(suspension_events)
    signals_df = pd.DataFrame(signal_log)

    metrics = calc_metrics(equity, initial_capital)
    metrics.update(
        {
            "start_date": str(run_days[0].date()),
            "end_date": str(run_days[-1].date()),
            "momentum_days": int(cfg.get("momentum_days", 10)),
            "top_n_sectors": int(cfg.get("top_n_sectors", 3)),
            "hold_days": hold_days,
            "staggered_portfolios": n_sleeves,
            "exclude_st": bool(cfg.get("exclude_st", False)),
            "st_detection": str(cfg.get("st_detection", "flag_or_name")),
            "sector_return_weighting": str(cfg.get("sector_return_weighting", "free_float_mcap")),
            "constituent_weighting": str(cfg.get("constituent_weighting", "free_float_mcap")),
            "trade_records": int(len(trades_df)),
            "buy_records": int((trades_df.get("side") == "BUY").sum()) if not trades_df.empty else 0,
            "sell_records": int((trades_df.get("side") == "SELL").sum()) if not trades_df.empty else 0,
            "total_fees": float(trades_df["fee"].sum()) if not trades_df.empty else 0.0,
            "total_taxes": float(trades_df["tax"].sum()) if not trades_df.empty else 0.0,
            "suspension_exit_events": int(len(susp_df)),
            "resolved_suspension_exits": int((susp_df.get("status") == "resolved").sum()) if not susp_df.empty else 0,
            "unresolved_frozen_positions": int(
                sum(sum(1 for p in s.positions if p.overdue) for s in sleeves)
            ),
            "ending_open_positions": int(sum(len(s.positions) for s in sleeves)),
        }
    )

    # Save outputs.
    equity.to_csv(out / "equity_curve.csv", index=False, encoding="utf-8-sig")
    sector_daily.to_csv(out / "sector_daily.csv", index=False, encoding="utf-8-sig")
    signals_df.to_csv(out / "sector_signals.csv", index=False, encoding="utf-8-sig")
    susp_df.to_csv(out / "suspension_events.csv", index=False, encoding="utf-8-sig")
    if bool(cfg.get("save_trade_log", True)):
        trades_df.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")

    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=json_default)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(equity["date"], equity["nav"], label="Strategy NAV")
    ax.set_title("Sector Momentum Rotation - Stock-level Reconstruction")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=160)
    plt.close(fig)

    print("\n===== Backtest summary =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nResults written to: {out}")

    if metrics["unresolved_frozen_positions"]:
        print(
            "WARNING: Some positions were still frozen at the end of the dataset. "
            "They were NOT force-liquidated."
        )


if __name__ == "__main__":
    main()
