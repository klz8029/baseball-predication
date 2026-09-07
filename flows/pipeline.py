"""Prefect flow that replaces the Windows scheduled task.

DAG per run:

    settle_and_snapshot ─┐
                         ├─> build_warehouse ─> refresh_dashboard_data
    ingest_recent_odds ──┘
    refresh_feature_cache        (independent; feeds the API container)
    track_experiments            (weekly; parametrised)

    prefect deploy flows/pipeline.py:daily_pipeline -n baseball-daily --cron "0 16,20,23 * * *"
    python flows/pipeline.py     # ad-hoc run
"""
from __future__ import annotations

import datetime as dt
import os
import sys

from prefect import flow, get_run_logger, task

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@task(retries=2, retry_delay_seconds=30)
def settle_and_snapshot() -> str:
    import forward
    forward.settle()
    forward.snapshot()
    return "forward_log.csv updated"


@task(retries=1)
def ingest_recent_odds(days_back: int = 10) -> str:
    """Pull the last `days_back` days of Kalshi settled odds. Skips cleanly with no key."""
    log = get_run_logger()
    try:
        import build_odds
        end = dt.date.today()
        start = end - dt.timedelta(days=days_back)
        out = os.path.join(ROOT, "data", "odds_incremental.csv")   # rolling file; warehouse dedups
        build_odds.build(str(start), str(end), out)
        return f"wrote {out}"
    except Exception as e:  # no key / offline / API change
        log.warning(f"odds ingest skipped: {e}")
        return "skipped"


@task
def refresh_feature_cache() -> str:
    from serving.build_cache import build
    build()
    return "data/feature_cache.pkl refreshed"


@task
def build_warehouse(_dep1: str, _dep2: str) -> dict:
    from warehouse.db import build
    return build(verbose=False)


@task
def refresh_dashboard_data(counts: dict) -> str:
    # marts are the dashboard's source; nothing extra to do, just report
    return f"warehouse rebuilt: {counts.get('mart_bets', 0)} bets, " \
           f"{counts.get('stg_games', 0)} games"


@task
def track_experiments() -> str:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from experiments.track import main
    main()
    return "mlflow runs logged"


@flow(name="baseball-daily")
def daily_pipeline(run_experiments: bool = False, fast: bool = False):
    """fast=True skips the two slow network tasks (forward loop, odds ingest) so a
    demo run finishes in ~1 min and only rebuilds the warehouse + feature cache."""
    fwd = "skipped(fast)" if fast else settle_and_snapshot()
    odds = "skipped(fast)" if fast else ingest_recent_odds()
    refresh_feature_cache()
    counts = build_warehouse(fwd, odds)
    msg = refresh_dashboard_data(counts)
    if run_experiments:
        track_experiments()
    get_run_logger().info(msg)
    return msg


if __name__ == "__main__":
    print(daily_pipeline(run_experiments="--experiments" in sys.argv,
                         fast="--fast" in sys.argv))
