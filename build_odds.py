"""Build an MLB game-odds dataset from Kalshi + MLB statsapi for a date range.

Mirrors the notebook's build_dataset() but parametrised. Writes a CSV with
columns: game_date, start_time_utc, away_team, home_team, away_score, home_score,
winner, away_implied_win_prob, home_implied_win_prob.

Credentials: KALSHI_KEY_ID from env (falls back to the Windows user registry),
private key from KALSHI_PRIVATE_KEY_PATH or ~/.kalshi/kalshi_private_key.pem.
"""
from __future__ import annotations

import base64
import datetime as dt
import os
import subprocess
import sys
from datetime import timedelta

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
KALSHI_API_BASE = "https://external-api.kalshi.com"

TEAM_TICKERS = {
    "Los Angeles Dodgers": "LAD", "New York Yankees": "NYY", "Chicago Cubs": "CHC",
    "Boston Red Sox": "BOS", "Houston Astros": "HOU", "Atlanta Braves": "ATL",
    "Philadelphia Phillies": "PHI", "Baltimore Orioles": "BAL", "Texas Rangers": "TEX",
    "Seattle Mariners": "SEA", "Toronto Blue Jays": "TOR", "Tampa Bay Rays": "TB",
    "Minnesota Twins": "MIN", "Cleveland Guardians": "CLE", "Chicago White Sox": "CWS",
    "Detroit Tigers": "DET", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA",
    "Oakland Athletics": "OAK", "Athletics": "OAK", "New York Mets": "NYM",
    "Washington Nationals": "WSH", "Miami Marlins": "MIA", "Pittsburgh Pirates": "PIT",
    "Cincinnati Reds": "CIN", "Milwaukee Brewers": "MIL", "St. Louis Cardinals": "STL",
    "Colorado Rockies": "COL", "Arizona Diamondbacks": "ARI", "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
}


def _key_id() -> str:
    kid = os.environ.get("KALSHI_KEY_ID")
    if kid:
        return kid
    try:  # Windows user-registry fallback (setx without a shell restart)
        return subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command",
             '[Environment]::GetEnvironmentVariable("KALSHI_KEY_ID","User")'],
            text=True).strip()
    except Exception:
        return ""


KALSHI_KEY_ID = _key_id()
PRIVATE_KEY_FILE = os.environ.get(
    "KALSHI_PRIVATE_KEY_PATH",
    os.path.join(os.path.expanduser("~"), ".kalshi", "kalshi_private_key.pem"))
with open(PRIVATE_KEY_FILE, "rb") as _f:
    _PK = serialization.load_pem_private_key(_f.read(), password=None)


def kalshi_headers(method: str, path: str) -> dict:
    ts = str(int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000))
    msg = f"{ts}{method.upper()}{path.split('?')[0]}".encode()
    sig = _PK.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                    salt_length=padding.PSS.DIGEST_LENGTH),
                   hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts, "Content-Type": "application/json"}


def mlb_games(start_date: str, end_date: str) -> list:
    url = f"{MLB_API_BASE}/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    games = []
    for day in r.json().get("dates", []):
        for g in day["games"]:
            if g.get("gameType") != "R" or g.get("status", {}).get("statusCode") != "F":
                continue
            t = dt.datetime.strptime(g["gameDate"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc)
            aw, hm = g["teams"]["away"]["team"]["name"], g["teams"]["home"]["team"]["name"]
            if "score" not in g["teams"]["away"] or "score" not in g["teams"]["home"]:
                continue
            asc, hsc = g["teams"]["away"]["score"], g["teams"]["home"]["score"]
            if aw in TEAM_TICKERS and hm in TEAM_TICKERS:
                games.append({"game_date": t.strftime("%Y-%m-%d"), "start_time_utc": t,
                              "away_team": aw, "home_team": hm,
                              "away_score": asc, "home_score": hsc,
                              "winner": aw if asc > hsc else hm,
                              "ticker_away": TEAM_TICKERS[aw], "ticker_home": TEAM_TICKERS[hm]})
    return games


def all_kalshi_mlb_tickers() -> list:
    tickers = []
    for base in ["/trade-api/v2/markets?series_ticker=KXMLBGAME&limit=1000",
                 "/trade-api/v2/historical/markets?series_ticker=KXMLBGAME&limit=1000"]:
        url = f"{KALSHI_API_BASE}{base}"
        while url:
            path = url.replace(KALSHI_API_BASE, "").split("?")[0]
            r = requests.get(url, headers=kalshi_headers("GET", path), timeout=30)
            if r.status_code != 200:
                break
            data = r.json()
            tickers += [m["ticker"] for m in data.get("markets", [])]
            cur = data.get("cursor")
            url = f"{KALSHI_API_BASE}{base}&cursor={cur}" if cur else None
    return tickers


def match_ticker(game: dict, tickers: list) -> str | None:
    d = game["start_time_utc"].strftime("%y%b%d").upper()
    aw, hm = game["ticker_away"], game["ticker_home"]
    prefix = f"KXMLBGAME-{d}"
    for t in tickers:
        if t.startswith(prefix) and aw in t and hm in t and t.endswith(f"-{aw}"):
            return t
    return None


SERIES = "KXMLBGAME"


def _candle_price(c: dict):
    """Away-team YES price from a candle: last trade close, else mean, else mid."""
    px = c.get("price", {})
    for k in ("close_dollars", "mean_dollars", "previous_dollars"):
        v = px.get(k)
        if v is not None and float(v) > 0:
            return float(v)
    bid = c.get("yes_bid", {}).get("close_dollars")
    ask = c.get("yes_ask", {}).get("close_dollars")
    if bid is not None and ask is not None and float(ask) > 0:
        return (float(bid) + float(ask)) / 2.0
    return None


def odds_1h_before(ticker: str, start_utc: dt.datetime):
    path = f"/trade-api/v2/series/{SERIES}/markets/{ticker}/candlesticks"
    params = {"period_interval": 60,
              "start_ts": int((start_utc - timedelta(hours=5)).timestamp()),
              "end_ts": int((start_utc - timedelta(minutes=45)).timestamp())}
    r = requests.get(f"{KALSHI_API_BASE}{path}",
                     headers=kalshi_headers("GET", path), params=params, timeout=30)
    if r.status_code != 200:
        return None
    for c in reversed(r.json().get("candlesticks", [])):
        px = _candle_price(c)
        if px is not None:
            return px
    return None


def build(start_date: str, end_date: str, out_file: str):
    print(f"key id: {KALSHI_KEY_ID[:8]}...   MLB {start_date}..{end_date}")
    games = mlb_games(start_date, end_date)
    tickers = all_kalshi_mlb_tickers()
    print(f"{len(games)} MLB games, {len(tickers)} Kalshi markets\n")

    rows = []
    for i, g in enumerate(games, 1):
        tk = match_ticker(g, tickers)
        if not tk:
            print(f"[{i}/{len(games)}] no market: {g['away_team']} @ {g['home_team']} {g['game_date']}")
            continue
        px = odds_1h_before(tk, g["start_time_utc"])
        if px is None:
            print(f"[{i}/{len(games)}] no history: {tk}")
            continue
        g["away_implied_win_prob"] = round(px, 3)
        g["home_implied_win_prob"] = round(1.0 - px, 3)
        rows.append(g)
        print(f"[{i}/{len(games)}] {tk}  away {g['away_implied_win_prob']}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nno odds found"); return
    df = df.drop(columns=["ticker_away", "ticker_home"])
    df.to_csv(out_file, index=False)
    print(f"\nsaved {len(df)} games -> {out_file}")


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "2026-08-01"
    e = sys.argv[2] if len(sys.argv) > 2 else "2026-08-27"
    o = sys.argv[3] if len(sys.argv) > 3 else "MLB_2026_August_Odds_And_Results.csv"
    build(s, e, o)
