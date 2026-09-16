# Sample Data

Date range:

2025-11-01 to 2026-01-31

The actual trading-date coverage in the files is 2025-11-03 to 2026-01-30.

This directory contains a compact sample of the full A-share daily dataset.

The sample:

- contains all stocks appearing during the selected dates;
- retains ST stocks;
- retains suspended stocks;
- retains historical industry labels;
- does not clean or modify price data;
- only removes columns that are not required by the backtest.

The source monthly files contain 89 columns. The sample keeps the 20 fields required by `prepare_data.py` and `run_backtest.py`. Each selected field value was checked after writing and reading the sample files back; the per-month value hashes matched.

## Files

- `2025-11.csv` — 108,992 rows, 18,486,147 bytes
- `2025-12.csv` — 125,582 rows, 21,287,213 bytes
- `2026-01.csv` — 109,439 rows, 18,628,744 bytes

## Statistics

- min_date: `2025-11-03`
- max_date: `2026-01-30`
- trading_days: `63`
- rows: `344,013`
- unique_stocks: `5,482`
- industries: `31` nonblank industry labels
- suspended_rows: `797`
- ST_rows: `11,200` rows where the raw `是否ST` field is nonzero; the prepared cache reports `11,274` rows under `flag_or_name` detection, with 210 flag/name disagreement rows
- total_size: `58,402,104` bytes (about 55.72 MiB)

## Columns

1. `股票代码`
2. `股票简称`
3. `行业`
4. `日期`
5. `开盘价`
6. `最高价`
7. `最低价`
8. `收盘价`
9. `成交量（股）`
10. `成交额（元）`
11. `换手率`
12. `前收盘价`
13. `涨停价`
14. `跌停价`
15. `复权因子`
16. `总市值`
17. `流通市值`
18. `自由流通市值`
19. `是否停牌`
20. `是否ST`
