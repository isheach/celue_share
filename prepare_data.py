from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import duckdb
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sql_literal(text: str) -> str:
    return text.replace("'", "''")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a compact DuckDB cache from A-share daily CSV files.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"])
    db_path = Path(cfg.get("cache_db", "./cache/market.duckdb"))

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    csv_files = sorted(data_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {data_root}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        if not args.force:
            print(f"Cache already exists: {db_path}")
            print("Use --force to rebuild it.")
            return
        db_path.unlink()

    csv_glob = str((data_root / "*.csv").resolve()).replace("\\", "/")
    csv_glob_sql = sql_literal(csv_glob)

    con = duckdb.connect(str(db_path))
    con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 4) - 1)}")
    con.execute("PRAGMA enable_progress_bar=true")

    print(f"Found {len(csv_files)} CSV files")
    print(f"Source: {data_root}")
    print(f"Cache : {db_path}")

    con.execute(
        f"""
        CREATE TABLE daily_stage AS
        SELECT
            NULLIF(TRIM("股票代码"), '') AS code,
            NULLIF(TRIM("股票简称"), '') AS name,
            NULLIF(TRIM("行业"), '') AS industry,
            TRY_CAST("日期" AS DATE) AS date,
            TRY_CAST(NULLIF("开盘价", '') AS DOUBLE) AS open,
            TRY_CAST(NULLIF("最高价", '') AS DOUBLE) AS high,
            TRY_CAST(NULLIF("最低价", '') AS DOUBLE) AS low,
            TRY_CAST(NULLIF("收盘价", '') AS DOUBLE) AS close,
            TRY_CAST(NULLIF("成交量（股）", '') AS DOUBLE) AS volume,
            TRY_CAST(NULLIF("成交额（元）", '') AS DOUBLE) AS amount,
            TRY_CAST(NULLIF("换手率", '') AS DOUBLE) AS turnover,
            TRY_CAST(NULLIF("前收盘价", '') AS DOUBLE) AS pre_close,
            TRY_CAST(NULLIF("涨停价", '') AS DOUBLE) AS limit_up,
            TRY_CAST(NULLIF("跌停价", '') AS DOUBLE) AS limit_down,
            TRY_CAST(NULLIF("复权因子", '') AS DOUBLE) AS adj_factor,
            TRY_CAST(NULLIF("总市值", '') AS DOUBLE) AS total_mcap,
            TRY_CAST(NULLIF("流通市值", '') AS DOUBLE) AS float_mcap,
            TRY_CAST(NULLIF("自由流通市值", '') AS DOUBLE) AS free_float_mcap,
            COALESCE(TRY_CAST(NULLIF("是否停牌", '') AS DOUBLE), 0.0) AS suspended_raw,
            COALESCE(TRY_CAST(NULLIF("是否ST", '') AS DOUBLE), 0.0) AS st_raw,
            COALESCE(TRY_CAST(NULLIF("是否停牌", '') AS DOUBLE), 0.0) <> 0.0 AS is_suspended,
            COALESCE(TRY_CAST(NULLIF("是否ST", '') AS DOUBLE), 0.0) <> 0.0 AS is_st_flag,
            UPPER(COALESCE("股票简称", '')) LIKE '%ST%' AS is_st_name,
            (
                COALESCE(TRY_CAST(NULLIF("是否ST", '') AS DOUBLE), 0.0) <> 0.0
                OR UPPER(COALESCE("股票简称", '')) LIKE '%ST%'
            ) AS is_st,
            CASE
                WHEN TRY_CAST(NULLIF("收盘价", '') AS DOUBLE) > 0
                 AND TRY_CAST(NULLIF("前收盘价", '') AS DOUBLE) > 0
                THEN TRY_CAST(NULLIF("收盘价", '') AS DOUBLE)
                     / TRY_CAST(NULLIF("前收盘价", '') AS DOUBLE) - 1.0
                ELSE NULL
            END AS stock_return
        FROM read_csv_auto(
            '{csv_glob_sql}',
            header=true,
            all_varchar=true,
            union_by_name=true,
            quote='"',
            escape='"',
            ignore_errors=false
        )
        WHERE NULLIF(TRIM("股票代码"), '') IS NOT NULL
          AND TRY_CAST("日期" AS DATE) IS NOT NULL
        """
    )

    con.execute(
        """
        CREATE TABLE daily AS
        SELECT
            *,
            LAG(total_mcap) OVER (PARTITION BY code ORDER BY date) AS prev_total_mcap,
            LAG(float_mcap) OVER (PARTITION BY code ORDER BY date) AS prev_float_mcap,
            LAG(free_float_mcap) OVER (PARTITION BY code ORDER BY date) AS prev_free_float_mcap
        FROM daily_stage
        ORDER BY date, code
        """
    )
    con.execute("DROP TABLE daily_stage")
    con.execute("CREATE TABLE trading_days AS SELECT DISTINCT date FROM daily ORDER BY date")
    con.execute(
        """
        CREATE TABLE metadata AS
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT code) AS stocks,
            COUNT(DISTINCT date) AS trading_days,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            SUM(CASE WHEN industry IS NULL THEN 1 ELSE 0 END) AS blank_industry_rows,
            SUM(CASE WHEN is_suspended THEN 1 ELSE 0 END) AS suspended_rows,
            SUM(CASE WHEN is_st THEN 1 ELSE 0 END) AS st_rows
        FROM daily
        """
    )
    try:
        con.execute("CREATE INDEX idx_daily_date ON daily(date)")
        con.execute("CREATE INDEX idx_daily_code_date ON daily(code, date)")
    except duckdb.Error as exc:
        print(f"Index creation skipped (non-fatal): {exc}")
    con.execute("ANALYZE daily")

    print("\nPrepared cache summary:")
    print(con.execute("SELECT * FROM metadata").fetchdf().to_string(index=False))
    disagree = con.execute("SELECT COUNT(*) FROM daily WHERE is_st_flag <> is_st_name").fetchone()[0]
    print(f"Rows where ST flag and stock-name ST detection disagree: {disagree:,}")
    con.close()
    print(f"\nDone: {db_path}")


if __name__ == "__main__":
    main()
