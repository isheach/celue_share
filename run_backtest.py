from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the sector-momentum backtest")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


@dataclass
class PositionLot:
    lot_id: str
    code: str
    name: str
    sector: str
    shares: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    scheduled_exit_date: pd.Timestamp
    entry_price: float
    last_price: float
    overdue: bool = False
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


def weighting_column(mode: str, lagged: bool = False) -> str | None:
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
    return series.replace(aliases) if aliases else series


def positive(v: Any) -> bool:
    return pd.notna(v) and float(v) > 0


def signal_eligible_mask(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    mask = (
        ~df["is_suspended"].fillna(True).astype(bool)
        & df["industry"].notna()
        & (df["close"].fillna(0) > 0)
        & (df["pre_close"].fillna(0) > 0)
    )
    if bool(cfg.get("require_positive_volume", True)):
        mask &= df["volume"].fillna(0) > 0
    if bool(cfg.get("exclude_st", False)):
        mask &= ~df[chosen_st_column(cfg)].fillna(False).astype(bool)
    return mask


def buyable(row: pd.Series, cfg: dict[str, Any]) -> bool:
    if bool(row.get("is_suspended", True)) or not positive(row.get("open")):
        return False
    if bool(cfg.get("require_positive_volume", True)) and not positive(row.get("volume")):
        return False
    if bool(cfg.get("exclude_st", False)) and bool(row.get(chosen_st_column(cfg), False)):
        return False
    return True


def sellable(row: pd.Series, price_col: str, cfg: dict[str, Any]) -> bool:
    if bool(row.get("is_suspended", True)) or not positive(row.get(price_col)):
        return False
    if bool(cfg.get("require_positive_volume", True)) and not positive(row.get("volume")):
        return False
    return True


def member_weights(members: pd.DataFrame, mode: str) -> dict[str, float]:
    if members.empty:
        return {}
    if mode.lower() == "equal":
        w = 1.0 / len(members)
        return {str(c): w for c in members["code"]}
    col = weighting_column(mode, lagged=False)
    assert col is not None
    vals = pd.to_numeric(members[col], errors="coerce")
    valid = vals.notna() & (vals > 0)
    # Preserve all eligible members. If one cap is missing, fall back to equal.
    if valid.sum() != len(members) or float(vals[valid].sum()) <= 0:
        w = 1.0 / len(members)
        return {str(c): w for c in members["code"]}
    total = float(vals.sum())
    return {str(c): float(v) / total for c, v in zip(members["code"], vals, strict=False)}


def round_shares(raw: float, cfg: dict[str, Any]) -> float:
    if bool(cfg.get("allow_fractional_shares", True)):
        return max(0.0, raw)
    lot = int(cfg.get("lot_size", 100))
    return float(max(0, math.floor(raw / lot) * lot))


def build_sector_signals(
    con: duckdb.DuckDBPyConnection, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[pd.Timestamp, list[tuple[str, float]]]]:
    st_col = chosen_st_column(cfg)
    clauses = ["industry IS NOT NULL", "NOT is_suspended", "stock_return IS NOT NULL"]
    if bool(cfg.get("require_positive_volume", True)):
        clauses.append("volume > 0")
    if bool(cfg.get("exclude_st", False)):
        clauses.append(f"NOT {st_col}")
    where_sql = " AND ".join(clauses)

    mode = str(cfg.get("sector_return_weighting", "free_float_mcap")).lower()
    wcol = weighting_column(mode, lagged=True)
    if wcol is None:
        sql = f"""
        SELECT date, industry, COUNT(*) n_constituents,
               AVG(stock_return) sector_return, FALSE cap_weight_fallback
        FROM daily WHERE {where_sql}
        GROUP BY date, industry ORDER BY date, industry
        """
    else:
        sql = f"""
        WITH g AS (
          SELECT date, industry, COUNT(*) n_constituents,
                 SUM(CASE WHEN {wcol} > 0 THEN 1 ELSE 0 END) valid_n,
                 SUM(CASE WHEN {wcol} > 0 THEN {wcol} ELSE 0 END) wsum,
                 SUM(CASE WHEN {wcol} > 0 THEN stock_return * {wcol} ELSE 0 END) wrsum,
                 AVG(stock_return) eret
          FROM daily WHERE {where_sql}
          GROUP BY date, industry
        )
        SELECT date, industry, n_constituents,
               CASE WHEN valid_n=n_constituents AND wsum>0 THEN wrsum/wsum ELSE eret END sector_return,
               CASE WHEN valid_n=n_constituents AND wsum>0 THEN FALSE ELSE TRUE END cap_weight_fallback
        FROM g ORDER BY date, industry
        """
    sector = con.execute(sql).fetchdf()
    sector["date"] = pd.to_datetime(sector["date"]).dt.normalize()
    sector["industry"] = apply_industry_aliases(sector["industry"], cfg)

    # Optional alias merging. Default config leaves labels untouched.
    if (cfg.get("industry_aliases") or {}) and sector.duplicated(["date", "industry"]).any():
        sector["x"] = sector["sector_return"] * sector["n_constituents"]
        sector = sector.groupby(["date", "industry"], as_index=False).agg(
            n_constituents=("n_constituents", "sum"),
            x=("x", "sum"),
            cap_weight_fallback=("cap_weight_fallback", "max"),
        )
        sector["sector_return"] = sector["x"] / sector["n_constituents"]
        sector.drop(columns="x", inplace=True)

    days = con.execute("SELECT date FROM trading_days ORDER BY date").fetchdf()
    all_days = pd.DatetimeIndex(pd.to_datetime(days["date"]).dt.normalize())
    ret = sector.pivot(index="date", columns="industry", values="sector_return").reindex(all_days)
    n = sector.pivot(index="date", columns="industry", values="n_constituents").reindex(index=all_days, columns=ret.columns)

    window = int(cfg.get("momentum_days", 10))
    momentum = (1.0 + ret).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
    top_n = int(cfg.get("top_n_sectors", 3))
    min_n = int(cfg.get("min_sector_constituents", 1))
    signals: dict[pd.Timestamp, list[tuple[str, float]]] = {}
    for dt in all_days:
        row = momentum.loc[dt].dropna()
        eligible = n.loc[dt][n.loc[dt] >= min_n].index
        top = row[row.index.isin(eligible)].sort_values(ascending=False).head(top_n)
        signals[normalize_ts(dt)] = [(str(k), float(v)) for k, v in top.items()]

    mom_long = momentum.stack(dropna=False, future_stack=False).rename("momentum").reset_index()
    mom_long.columns = ["date", "industry", "momentum"]
    sector = sector.merge(mom_long, on=["date", "industry"], how="left")
    return sector, all_days, signals


def fetch_day(con: duckdb.DuckDBPyConnection, dt: pd.Timestamp, cfg: dict[str, Any]) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT code,name,industry,date,open,high,low,close,volume,amount,pre_close,
               limit_up,limit_down,total_mcap,float_mcap,free_float_mcap,
               is_suspended,is_st_flag,is_st_name,is_st
        FROM daily WHERE date=? ORDER BY code
        """,
        [dt.date()],
    ).fetchdf()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["industry"] = apply_industry_aliases(df["industry"], cfg)
    return df


def sell_lot(sleeve: Sleeve, lot: PositionLot, price: float, dt: pd.Timestamp, execution: str,
             cfg: dict[str, Any], trades: list[dict[str, Any]]) -> None:
    gross = lot.shares * price
    fee = gross * float(cfg.get("sell_fee_rate", 0.0003))
    tax = gross * float(cfg.get("sell_tax_rate", 0.001))
    sleeve.cash += gross - fee - tax
    trades.append({
        "date": dt, "sleeve": sleeve.sleeve_id, "side": "SELL", "execution": execution,
        "code": lot.code, "name": lot.name, "sector": lot.sector, "shares": lot.shares,
        "price": price, "gross_notional": gross, "fee": fee, "tax": tax,
        "signal_date": lot.signal_date, "entry_date": lot.entry_date,
        "scheduled_exit_date": lot.scheduled_exit_date,
    })
    sleeve.positions.remove(lot)


def overdue_open_exits(sleeve: Sleeve, day: pd.DataFrame, dt: pd.Timestamp, cfg: dict[str, Any],
                       trades: list[dict[str, Any]], events: list[dict[str, Any]],
                       day_index: dict[pd.Timestamp, int]) -> None:
    if day.empty:
        return
    rows = day.set_index("code", drop=False)
    for lot in list(sleeve.positions):
        if not lot.overdue or lot.code not in rows.index:
            continue
        row = rows.loc[lot.code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if not sellable(row, "open", cfg):
            continue
        px = float(row["open"])
        sell_lot(sleeve, lot, px, dt, "OPEN_AFTER_SUSPENSION", cfg, trades)
        if lot.freeze_event_index is not None:
            e = events[lot.freeze_event_index]
            e.update(status="resolved", actual_exit_date=dt, actual_exit_price=px,
                     delay_trading_days=day_index[dt] - day_index[lot.scheduled_exit_date])


def execute_entry(sleeve: Sleeve, day: pd.DataFrame, dt: pd.Timestamp, cfg: dict[str, Any],
                  trades: list[dict[str, Any]], lot_counter: list[int]) -> None:
    p = sleeve.pending
    if p is None or p.entry_date != dt:
        return
    if day.empty:
        sleeve.pending = None
        return
    rows = day.set_index("code", drop=False)
    starting_cash = sleeve.cash
    top_n = int(cfg.get("top_n_sectors", 3))
    buy_fee = float(cfg.get("buy_fee_rate", 0.0003))

    for selected in p.selected_sectors:
        sector = selected["sector"]
        sig_w: dict[str, float] = selected["member_weights"]
        codes = []
        for code in sig_w:
            if code not in rows.index:
                continue
            row = rows.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if buyable(row, cfg):
                codes.append(code)
        if not codes:
            continue
        wsum = sum(sig_w[c] for c in codes)
        if wsum <= 0:
            continue
        sector_budget = starting_cash / top_n
        for code in codes:
            row = rows.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            px = float(row["open"])
            budget = sector_budget * sig_w[code] / wsum
            shares = round_shares(budget / (px * (1 + buy_fee)), cfg)
            if shares <= 0:
                continue
            gross = shares * px
            fee = gross * buy_fee
            cost = gross + fee
            if cost > sleeve.cash + 1e-8:
                shares = round_shares(sleeve.cash / (px * (1 + buy_fee)), cfg)
                if shares <= 0:
                    continue
                gross = shares * px
                fee = gross * buy_fee
                cost = gross + fee
            sleeve.cash -= cost
            lot_counter[0] += 1
            lot = PositionLot(
                lot_id=f"S{sleeve.sleeve_id}-{lot_counter[0]}", code=str(code),
                name=str(row.get("name") or ""), sector=sector, shares=shares,
                signal_date=p.signal_date, entry_date=dt, scheduled_exit_date=p.scheduled_exit_date,
                entry_price=px, last_price=px,
            )
            sleeve.positions.append(lot)
            trades.append({
                "date": dt, "sleeve": sleeve.sleeve_id, "side": "BUY", "execution": "OPEN",
                "code": lot.code, "name": lot.name, "sector": sector, "shares": shares,
                "price": px, "gross_notional": gross, "fee": fee, "tax": 0.0,
                "signal_date": p.signal_date, "entry_date": dt,
                "scheduled_exit_date": p.scheduled_exit_date,
            })
    sleeve.pending = None


def scheduled_close_exits(sleeve: Sleeve, day: pd.DataFrame, dt: pd.Timestamp,
                          cfg: dict[str, Any], trades: list[dict[str, Any]],
                          events: list[dict[str, Any]]) -> None:
    rows = day.set_index("code", drop=False) if not day.empty else None
    for lot in list(sleeve.positions):
        if lot.overdue or lot.scheduled_exit_date != dt:
            continue
        row = None
        if rows is not None and lot.code in rows.index:
            row = rows.loc[lot.code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        if row is not None and sellable(row, "close", cfg):
            sell_lot(sleeve, lot, float(row["close"]), dt, "SCHEDULED_CLOSE", cfg, trades)
        else:
            lot.overdue = True
            lot.freeze_event_index = len(events)
            events.append({
                "sleeve": sleeve.sleeve_id, "code": lot.code, "name": lot.name,
                "sector": lot.sector, "entry_date": lot.entry_date,
                "scheduled_exit_date": lot.scheduled_exit_date, "freeze_start_date": dt,
                "status": "frozen", "actual_exit_date": None, "actual_exit_price": None,
                "delay_trading_days": None,
            })


def refresh_close_prices(sleeve: Sleeve, day: pd.DataFrame) -> None:
    if day.empty:
        return
    rows = day.set_index("code", drop=False)
    for lot in sleeve.positions:
        if lot.code not in rows.index:
            continue
        row = rows.loc[lot.code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if positive(row.get("close")):
            lot.last_price = float(row["close"])


def sleeve_equity(sleeve: Sleeve) -> float:
    return sleeve.cash + sum(l.shares * l.last_price for l in sleeve.positions)


def make_selected_sectors(day: pd.DataFrame, signal: list[tuple[str, float]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if day.empty:
        return []
    eligible = day[signal_eligible_mask(day, cfg)].copy()
    out: list[dict[str, Any]] = []
    mode = str(cfg.get("constituent_weighting", "free_float_mcap"))
    for sector, momentum in signal:
        members = eligible[eligible["industry"] == sector]
        w = member_weights(members, mode)
        if w:
            out.append({"sector": sector, "momentum": momentum, "member_weights": w})
    return out


def metrics(eq: pd.DataFrame, initial_capital: float) -> dict[str, float]:
    x = eq.copy()
    x["nav"] = x["total_equity"] / initial_capital
    x["daily_return"] = x["nav"].pct_change().fillna(0.0)
    x["drawdown"] = x["nav"] / x["nav"].cummax() - 1.0
    n = max(1, len(x) - 1)
    final_nav = float(x["nav"].iloc[-1])
    annual = final_nav ** (252.0 / n) - 1.0
    r = x["daily_return"].iloc[1:]
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252)) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0
    return {
        "final_nav": final_nav,
        "total_return": final_nav - 1.0,
        "annualized_return": float(annual),
        "max_drawdown": float(x["drawdown"].min()),
        "sharpe_0rf": sharpe,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    db_path = Path(cfg.get("cache_db", "./cache/market.duckdb"))
    if not db_path.exists():
        raise FileNotFoundError(f"Cache not found: {db_path}. Run prepare_data.py first.")

    out_root = Path(cfg.get("output_dir", "./outputs"))
    out = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    con = duckdb.connect(str(db_path), read_only=True)
    sector_daily, all_days, signals = build_sector_signals(con, cfg)
    day_index = {normalize_ts(d): i for i, d in enumerate(all_days)}

    start = normalize_ts(cfg.get("signal_start_date", all_days[0]))
    end_cfg = cfg.get("end_date")
    end = normalize_ts(end_cfg) if end_cfg else normalize_ts(all_days[-1])
    active_days = [normalize_ts(d) for d in all_days if start <= normalize_ts(d) <= end]
    if not active_days:
        raise ValueError("No trading days inside configured backtest range")

    initial = float(cfg.get("initial_capital", 1_000_000.0))
    sleeve_n = int(cfg.get("staggered_portfolios", 2))
    hold_days = int(cfg.get("hold_days", 2))
    sleeves = [Sleeve(i, initial / sleeve_n) for i in range(sleeve_n)]
    trades: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    lot_counter = [0]

    first_signal_index: int | None = None
    for dt in active_days:
        i = day_index[dt]
        day = fetch_day(con, dt, cfg)

        for s in sleeves:
            overdue_open_exits(s, day, dt, cfg, trades, events, day_index)
        for s in sleeves:
            execute_entry(s, day, dt, cfg, trades, lot_counter)
        for s in sleeves:
            scheduled_close_exits(s, day, dt, cfg, trades, events)
            refresh_close_prices(s, day)

        sleeve_vals = [sleeve_equity(s) for s in sleeves]
        row = {"date": dt, "total_equity": float(sum(sleeve_vals))}
        for j, val in enumerate(sleeve_vals):
            row[f"sleeve_{j}_equity"] = float(val)
        row["cash"] = float(sum(s.cash for s in sleeves))
        row["open_positions"] = int(sum(len(s.positions) for s in sleeves))
        row["frozen_positions"] = int(sum(sum(l.overdue for l in s.positions) for s in sleeves))
        curve.append(row)

        sig = signals.get(dt, [])
        # Need a next open and a scheduled exit day.
        if sig and i + hold_days < len(all_days) and i + 1 < len(all_days):
            if first_signal_index is None:
                first_signal_index = i
            sid = (i - first_signal_index) % sleeve_n
            sleeve = sleeves[sid]
            selected = make_selected_sectors(day, sig, cfg)
            if selected:
                sleeve.pending = PendingOrder(
                    signal_date=dt,
                    entry_date=normalize_ts(all_days[i + 1]),
                    scheduled_exit_date=normalize_ts(all_days[i + hold_days]),
                    selected_sectors=selected,
                )
                for rank, item in enumerate(selected, 1):
                    signal_rows.append({
                        "date": dt, "sleeve": sid, "rank": rank,
                        "industry": item["sector"], "momentum": item["momentum"],
                        "members": len(item["member_weights"]),
                    })

    con.close()
    eq = pd.DataFrame(curve)
    m = metrics(eq, initial)
    eq["nav"] = eq["total_equity"] / initial
    eq["daily_return"] = eq["nav"].pct_change().fillna(0.0)
    eq["drawdown"] = eq["nav"] / eq["nav"].cummax() - 1.0
    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(events)
    sdf = pd.DataFrame(signal_rows)

    total_fees = float(tdf["fee"].sum()) if not tdf.empty else 0.0
    total_taxes = float(tdf["tax"].sum()) if not tdf.empty else 0.0
    unresolved = int(sum(sum(l.overdue for l in s.positions) for s in sleeves))
    resolved = int((edf["status"] == "resolved").sum()) if not edf.empty else 0
    summary = {
        "start_date": str(active_days[0].date()),
        "end_date": str(active_days[-1].date()),
        "initial_capital": initial,
        **m,
        "trade_records": int(len(tdf)),
        "buy_records": int((tdf["side"] == "BUY").sum()) if not tdf.empty else 0,
        "sell_records": int((tdf["side"] == "SELL").sum()) if not tdf.empty else 0,
        "total_fees": total_fees,
        "total_taxes": total_taxes,
        "total_transaction_cost": total_fees + total_taxes,
        "suspension_exit_events": int(len(edf)),
        "resolved_suspension_exits": resolved,
        "unresolved_frozen_positions": unresolved,
    }

    eq.to_csv(out / "equity_curve.csv", index=False, encoding="utf-8-sig")
    sector_daily.to_csv(out / "sector_daily.csv", index=False, encoding="utf-8-sig")
    sdf.to_csv(out / "sector_signals.csv", index=False, encoding="utf-8-sig")
    edf.to_csv(out / "suspension_events.csv", index=False, encoding="utf-8-sig")
    if bool(cfg.get("save_trade_log", True)):
        tdf.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(eq["date"], eq["nav"])
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title("Sector Momentum Rotation")
    plt.tight_layout()
    plt.savefig(out / "equity_curve.png", dpi=150)
    plt.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
