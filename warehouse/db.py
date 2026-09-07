"""DuckDB warehouse: connection helper + build routine.

The warehouse file (data/warehouse.duckdb) holds:
  raw_odds, raw_wf_predictions, raw_forward_log   -- landed source data
  stg_games, stg_predictions                       -- cleaned, one grain each
  mart_bets, mart_calibration, mart_performance, mart_daily  -- analytics

`build()` lands the CSVs and runs warehouse/transforms/*.sql in filename order.
"""
from __future__ import annotations

import glob
import os

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "warehouse.duckdb")
TRANSFORMS = os.path.join(os.path.dirname(__file__), "transforms")

RAW_SOURCES = {
    # monthly backfill CSVs + the flow's rolling incremental file; stg_games
    # dedups to one row per (date, away, home)
    "raw_odds": (sorted(glob.glob(os.path.join(ROOT, "MLB_2026_*_Odds_And_Results.csv")))
                 + [os.path.join(ROOT, "data", "odds_incremental.csv")]),
    "raw_wf_predictions": [os.path.join(ROOT, "wf_predictions.csv")],
    "raw_forward_log": [os.path.join(ROOT, "forward_log.csv")],
}


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH, read_only=read_only)


def _land(con: duckdb.DuckDBPyConnection, table: str, files: list[str]) -> int:
    files = [f for f in files if os.path.exists(f)]
    if not files:
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT NULL LIMIT 0")
        return 0
    lst = ", ".join(f"'{f.replace(chr(92), '/')}'" for f in files)
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * FROM read_csv_auto([{lst}], union_by_name=true, all_varchar=true)
    """)
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def build(verbose: bool = True) -> dict:
    counts = {}
    with connect() as con:
        for table, files in RAW_SOURCES.items():
            counts[table] = _land(con, table, files)
        for path in sorted(glob.glob(os.path.join(TRANSFORMS, "*.sql"))):
            con.execute(open(path).read())
            name = os.path.basename(path)
            if verbose:
                print(f"  ran {name}")
        for t in ("stg_games", "stg_predictions", "mart_bets"):
            counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    if verbose:
        print("row counts:", counts)
    return counts


if __name__ == "__main__":
    build()
