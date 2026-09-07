"""Monte Carlo inning simulator, packaged so the eval harness can score it.

Pipeline for one game:
    team / lineup rates  --Log5-->  per-PA event probs for each side
    + park factor, ROE, times-through-order, bullpen phase
    event probs          --sim-->   half-inning run distribution (probabilistic
                                    base-running, GIDP, sac fly, steals)
    3 phase pmfs         --conv-->  9-inning team score distribution
    two score pmfs       --exact--> P(home win), ghost-runner extras included

`predict_games` knobs (all optional):
    window / shrink_pa       trailing-N-game team rates instead of season
    lineups / player_rates   offense from today's 9 hitters
    starters / starter_strength   opposing-starter ERA scale
    rest / rest_strength     short-rest starter penalty
    home_field               symmetric home offense multiplier (default 0.024)
    platoon                  team vs-LHP / vs-RHP offense multiplier
    realism                  master switch for the 5 realism upgrades below;
      park=..                per-park HR / hit multipliers          (gap 1)
      baserunning=..         probabilistic advancement + GIDP + SF + SB  (gap 2)
      bullpen=.. / tto=..    starter innings 1-6 (2nd/3rd time through) vs
                             a fresh bullpen for 7-9                 (gaps 3, 5-iid)
      ghost_runner=..        runner on 2nd to start each extra frame (gap 4)
      roe=..                 reached-on-error folded back in as ~singles

Event index order everywhere: [OUT, 1B, 2B, 3B, HR, BB].
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import requests

API = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}
EVENTS = ["OUT", "1B", "2B", "3B", "HR", "BB"]
_HITTING_KEYS = ["plateAppearances", "baseOnBalls", "hitByPitch", "hits",
                 "doubles", "triples", "homeRuns",
                 "strikeOuts", "groundOuts", "airOuts"]
_COUNT_COLS = ["pa", "bb", "hbp", "h", "d2", "d3", "hr"]

# ---- base-running advancement probabilities (league-average, from public data) --
ADV = {
    "1B_1to3": 0.28,   # runner on 1st takes 3rd on a single (else stops at 2nd)
    "1B_2toH": 0.62,   # runner on 2nd scores on a single    (else stops at 3rd)
    "2B_1toH": 0.42,   # runner on 1st scores on a double     (else stops at 3rd)
    "GO_3toH": 0.25,   # runner on 3rd scores on a groundout, <2 out
    "GO_2to3": 0.30,   # runner on 2nd -> 3rd on a groundout
    "GO_1to2": 0.35,   # runner on 1st -> 2nd on a groundout (fielder's choice-ish)
    "GIDP":    0.12,   # groundout, runner on 1st, <2 out -> double play
    "SF":      0.48,   # airout, runner on 3rd, <2 out -> run scores (sac fly)
    "AO_2to3": 0.12,   # runner on 2nd tags to 3rd on a fly, <2 out
    "SB_ATT":  0.06,   # per PA with runner on 1st & 2nd empty -> steal attempt
    "SB_OK":   0.76,   # steal success rate
}

# ---- rough 3-year park factors: (HR multiplier, hit/BABIP multiplier) ----------
# keyed by the HOME team's name; applied to BOTH offenses in that game.
PARK_FACTORS = {
    "Colorado Rockies": (1.18, 1.10), "Boston Red Sox": (1.03, 1.06),
    "Cincinnati Reds": (1.15, 1.02), "New York Yankees": (1.10, 1.00),
    "Philadelphia Phillies": (1.08, 1.00), "Milwaukee Brewers": (1.06, 1.00),
    "Chicago White Sox": (1.05, 1.00), "Texas Rangers": (1.04, 1.02),
    "Baltimore Orioles": (1.02, 1.00), "Atlanta Braves": (1.04, 1.00),
    "Arizona Diamondbacks": (1.03, 1.02), "Toronto Blue Jays": (1.02, 1.00),
    "Los Angeles Dodgers": (1.04, 0.96), "Houston Astros": (1.02, 1.00),
    "Athletics": (1.06, 1.03), "Tampa Bay Rays": (1.03, 1.00),
    "Chicago Cubs": (1.00, 1.00), "Washington Nationals": (1.00, 1.00),
    "Minnesota Twins": (1.00, 1.00), "Los Angeles Angels": (1.00, 0.99),
    "St. Louis Cardinals": (0.96, 1.00), "Cleveland Guardians": (0.98, 0.98),
    "Pittsburgh Pirates": (0.94, 1.00), "Kansas City Royals": (0.95, 1.02),
    "Detroit Tigers": (0.95, 1.00), "New York Mets": (0.95, 0.97),
    "Miami Marlins": (0.95, 0.95), "San Diego Padres": (0.92, 0.96),
    "Seattle Mariners": (0.90, 0.94), "San Francisco Giants": (0.90, 0.98),
}


# ----------------------------------------------------------------------------
# 1. rates
# ----------------------------------------------------------------------------
def _out_mix(stat: dict) -> tuple:
    """(K, groundout, airout) fractions of the in-play/K out bucket."""
    k = stat.get("strikeOuts", 0) or 0
    go = stat.get("groundOuts", 0) or 0
    ao = stat.get("airOuts", 0) or 0
    tot = k + go + ao
    if tot <= 0:
        return (0.33, 0.34, 0.33)
    return (k / tot, go / tot, ao / tot)


def get_rates(stat: dict) -> dict:
    """Per-PA (or per-BF) event rates plus '_mix' = (K, GO, AO) fractions."""
    pa = stat.get("plateAppearances") or stat.get("battersFaced")
    walks = stat["baseOnBalls"] + stat["hitByPitch"]
    singles = stat["hits"] - (stat["doubles"] + stat["triples"] + stat["homeRuns"])
    return {"1B": singles / pa, "2B": stat["doubles"] / pa, "3B": stat["triples"] / pa,
            "HR": stat["homeRuns"] / pa, "BB": walks / pa, "_mix": _out_mix(stat)}


def _rates_from_counts(c) -> dict:
    """Rates from raw counts pa,bb,hbp,h,d2,d3,hr (+ optional k,go,ao)."""
    pa = float(c["pa"])
    walks = c["bb"] + c["hbp"]
    singles = c["h"] - (c["d2"] + c["d3"] + c["hr"])
    mix = _out_mix({"strikeOuts": c.get("k", 0), "groundOuts": c.get("go", 0),
                    "airOuts": c.get("ao", 0)})
    return {"1B": singles / pa, "2B": c["d2"] / pa, "3B": c["d3"] / pa,
            "HR": c["hr"] / pa, "BB": walks / pa, "_mix": mix}


def league_average_rates(splits: list[dict]) -> dict:
    tot = {k: 0 for k in _HITTING_KEYS}
    for t in splits:
        for k in _HITTING_KEYS:
            tot[k] += t["stat"].get(k, 0)
    return get_rates(tot)


def matchup_probs(bat: dict, pit: dict, lg: dict, pitch_scale: float = 1.0) -> np.ndarray:
    """Log5 blend -> [OUT,1B,2B,3B,HR,BB], sums to 1."""
    ev = np.array([(bat[e] * pit[e] * pitch_scale) / lg[e]
                   for e in ("1B", "2B", "3B", "HR", "BB")])
    ev = np.clip(ev, 0.0, None)
    out = 1.0 - ev.sum()
    probs = np.concatenate([[out], ev])
    if out < 0:
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()
    return probs


def blend_out_mix(bat: dict, pit: dict, w_bat: float = 0.6) -> tuple:
    """Blend batter and pitcher (K,GO,AO) fractions."""
    b, p = bat.get("_mix", (.33, .34, .33)), pit.get("_mix", (.33, .34, .33))
    m = tuple(w_bat * bi + (1 - w_bat) * pi for bi, pi in zip(b, p))
    s = sum(m)
    return tuple(x / s for x in m)


def scale_reach(probs: np.ndarray, factor: float) -> np.ndarray:
    """Multiply the 5 reach events by `factor`, let OUT absorb, renormalise."""
    p = probs.copy()
    p[1:] *= factor
    p[0] = max(0.0, 1.0 - p[1:].sum())
    return p / p.sum()


def apply_park(probs: np.ndarray, pf: tuple) -> np.ndarray:
    """pf = (HR factor, hit factor). Scales HR and 1B/2B/3B, OUT absorbs."""
    hr_f, hit_f = pf
    p = probs.copy()
    p[1:4] *= hit_f
    p[4] *= hr_f
    p[0] = max(0.0, 1.0 - p[1:].sum())
    return p / p.sum()


def add_roe(probs: np.ndarray, roe: float) -> np.ndarray:
    """Fold reached-on-error in as extra singles taken out of the OUT bucket."""
    p = probs.copy()
    take = min(roe, p[0])
    p[0] -= take
    p[1] += take
    return p


def starter_scale(starter_era, staff_era, strength: float = 0.3,
                  clip=(0.6, 1.6)) -> float:
    if not starter_era or not staff_era:
        return 1.0
    ratio = starter_era / staff_era
    return float(np.clip(1.0 + strength * (ratio - 1.0), *clip))


def rest_scale(rest_days, strength: float = 1.0) -> float:
    if rest_days is None:
        return 1.0
    table = {0: 1.10, 1: 1.10, 2: 1.08, 3: 1.045, 4: 1.0,
             5: 1.0, 6: 1.005, 7: 1.01}
    return 1.0 + strength * (table.get(int(rest_days), 1.01) - 1.0)


LINEUP_SLOT_WEIGHTS = np.array([0.129, 0.126, 0.123, 0.120, 0.117,
                                0.113, 0.110, 0.107, 0.104])
LINEUP_SLOT_WEIGHTS = LINEUP_SLOT_WEIGHTS / LINEUP_SLOT_WEIGHTS.sum()


def lineup_offense_rates(pids: list[int], player_rates: dict, out_mix: tuple) -> dict:
    """Slot-weighted blend of 9 hitters' rates; out-mix comes from the team."""
    lg = player_rates["_league"]
    w = LINEUP_SLOT_WEIGHTS if len(pids) == 9 else np.full(len(pids), 1 / len(pids))
    acc = {e: 0.0 for e in ("1B", "2B", "3B", "HR", "BB")}
    for wi, pid in zip(w, pids):
        pr = player_rates.get(pid, lg)
        for e in acc:
            acc[e] += wi * pr[e]
    acc["_mix"] = out_mix
    return acc


# ----------------------------------------------------------------------------
# 2. simulate a half-inning, vectorised over `n` trials
# ----------------------------------------------------------------------------
def simulate_half_innings(probs, n: int, rng: np.random.Generator, *,
                          out_mix: tuple | None = None, baserunning: bool = True,
                          steals: bool = True,
                          start_bases: tuple = (0, 0, 0),
                          start_outs: int = 0) -> np.ndarray:
    """Runs scored in one half-inning, per trial. `out_mix`=(K,GO,AO) enables
    GIDP / sac fly; None treats every out as a strikeout. `baserunning=False`
    falls back to the old deterministic advancement."""
    probs = np.asarray(probs, dtype=float)
    cdf = np.cumsum(probs)
    pk, pgo, _ = out_mix if out_mix is not None else (1.0, 0.0, 0.0)

    outs = np.full(n, start_outs, dtype=np.int8)
    runs = np.zeros(n, dtype=np.int32)
    b1 = np.full(n, start_bases[0], dtype=np.int8)
    b2 = np.full(n, start_bases[1], dtype=np.int8)
    b3 = np.full(n, start_bases[2], dtype=np.int8)

    for _ in range(80):
        alive = outs < 3
        if not alive.any():
            break

        if steals:
            opp = alive & (b1 == 1) & (b2 == 0)
            att = opp & (rng.random(n) < ADV["SB_ATT"])
            ok = att & (rng.random(n) < ADV["SB_OK"])
            cs = att & ~ok
            b2 = np.where(ok, 1, b2); b1 = np.where(ok | cs, 0, b1)
            outs = np.where(cs, np.minimum(outs + 1, 3), outs)
            alive = outs < 3

        ev = np.minimum(np.searchsorted(cdf, rng.random(n)), 5)
        ev = np.where(alive, ev, 0)
        is_1b, is_2b, is_3b = ev == 1, ev == 2, ev == 3
        is_hr, is_bb = ev == 4, ev == 5
        is_out = ev == 0

        o1, o2, o3 = b1.copy(), b2.copy(), b3.copy()
        nb1, nb2, nb3 = b1.copy(), b2.copy(), b3.copy()
        radd = np.zeros(n, dtype=np.int32)
        oadd = np.zeros(n, dtype=np.int8)

        # ---- OUT: split into K / GO / AO -----------------------------------
        if is_out.any():
            u = rng.random(n)
            is_k = is_out & (u < pk)
            is_go = is_out & (u >= pk) & (u < pk + pgo)
            is_ao = is_out & (u >= pk + pgo)
            oadd = np.where(is_out, 1, oadd)

            if baserunning and out_mix is not None:
                lt2 = outs < 2
                # GIDP
                gidp = is_go & (o1 == 1) & lt2 & (rng.random(n) < ADV["GIDP"])
                oadd = np.where(gidp, 2, oadd)
                nb1 = np.where(gidp, 0, nb1)
                # plain groundout advancement
                gp = is_go & ~gidp & lt2
                r3 = gp & (o3 == 1) & (rng.random(n) < ADV["GO_3toH"])
                radd += r3.astype(np.int32)
                nb3 = np.where(r3, 0, nb3)
                r23 = gp & (o2 == 1) & (nb3 == 0) & (rng.random(n) < ADV["GO_2to3"])
                nb2 = np.where(r23, 0, nb2); nb3 = np.where(r23, 1, nb3)
                r12 = gp & (o1 == 1) & (nb2 == 0) & (rng.random(n) < ADV["GO_1to2"])
                nb1 = np.where(r12, 0, nb1); nb2 = np.where(r12, 1, nb2)
                # sac fly + tag from 2nd
                sf = is_ao & (o3 == 1) & lt2 & (rng.random(n) < ADV["SF"])
                radd += sf.astype(np.int32)
                nb3 = np.where(sf, 0, nb3)
                tag2 = is_ao & (o2 == 1) & lt2 & (nb3 == 0) & (rng.random(n) < ADV["AO_2to3"])
                nb2 = np.where(tag2, 0, nb2); nb3 = np.where(tag2, 1, nb3)

        # ---- BB -----------------------------------------------------------
        m = is_bb
        radd += (m & (o1 == 1) & (o2 == 1) & (o3 == 1)).astype(np.int32)
        nb3 = np.where(m, np.where((o1 == 1) & (o2 == 1), 1, o3), nb3)
        nb2 = np.where(m, np.where(o1 == 1, 1, o2), nb2)
        nb1 = np.where(m, 1, nb1)

        # ---- HR / 3B ----------------------------------------------------
        m = is_hr
        radd += m * (o1 + o2 + o3 + 1)
        nb1 = np.where(m, 0, nb1); nb2 = np.where(m, 0, nb2); nb3 = np.where(m, 0, nb3)
        m = is_3b
        radd += m * (o1 + o2 + o3)
        nb1 = np.where(m, 0, nb1); nb2 = np.where(m, 0, nb2); nb3 = np.where(m, 1, nb3)

        # ---- 2B -------------------------------------------------------
        m = is_2b
        if baserunning:
            r1_home = m & (o1 == 1) & (rng.random(n) < ADV["2B_1toH"])
        else:
            r1_home = np.zeros(n, dtype=bool)
        radd += (m * (o2 + o3)).astype(np.int32) + r1_home.astype(np.int32)
        nb1 = np.where(m, 0, nb1)
        nb2 = np.where(m, 1, nb2)
        nb3 = np.where(m, (m & (o1 == 1) & ~r1_home).astype(np.int8), nb3)

        # ---- 1B ------------------------------------------------------
        m = is_1b
        if baserunning:
            sc2 = m & (o2 == 1) & (rng.random(n) < ADV["1B_2toH"])
            to3_from2 = m & (o2 == 1) & ~sc2
            to3_from1 = m & (o1 == 1) & ~to3_from2 & (rng.random(n) < ADV["1B_1to3"])
        else:
            sc2 = m & (o2 == 1)
            to3_from2 = np.zeros(n, dtype=bool)
            to3_from1 = np.zeros(n, dtype=bool)
        to2_from1 = m & (o1 == 1) & ~to3_from1
        radd += (m & (o3 == 1)).astype(np.int32) + sc2.astype(np.int32)
        nb1 = np.where(m, 1, nb1)
        nb2 = np.where(m, to2_from1.astype(np.int8), nb2)
        nb3 = np.where(m, (to3_from2 | to3_from1).astype(np.int8), nb3)

        # ---- commit (only for trials alive at event time) ----------------
        runs = runs + np.where(alive, radd, 0)
        outs = np.minimum(outs + np.where(alive, oadd, 0), 3)
        b1 = np.where(alive, nb1, b1)
        b2 = np.where(alive, nb2, b2)
        b3 = np.where(alive, nb3, b3)

    return runs.astype(np.int64)


def half_inning_pmf(probs, n, rng, *, cap: int = 22, **kw) -> np.ndarray:
    runs = np.clip(simulate_half_innings(probs, n, rng, **kw), 0, cap)
    pmf = np.bincount(runs, minlength=cap + 1).astype(float)
    return pmf / pmf.sum()


# ----------------------------------------------------------------------------
# 3. P(home win)
# ----------------------------------------------------------------------------
def _convolve_all(pmfs: list[np.ndarray]) -> np.ndarray:
    out = pmfs[0].copy()
    for p in pmfs[1:]:
        out = np.convolve(out, p)
    return out


def home_win_probability(away_phases, home_phases, *, away_mix, home_mix,
                         baserunning: bool = True, ghost_runner: bool = True,
                         n: int = 40_000, rng: np.random.Generator | None = None,
                         phase_innings=(3, 3, 3)) -> float:
    """`*_phases` = [early, mid, late] 6-vectors (offense vs TTO1 / TTO2 / bullpen).
    `*_mix` = that team's (K,GO,AO) fractions -- one tuple, or a list of 3
    (per-phase, e.g. a groundball starter then a different bullpen mix)."""
    rng = rng or np.random.default_rng(0)

    def _mix3(m):
        return m if (m and isinstance(m[0], (tuple, list))) else [m, m, m]

    def team_score_pmf(phases, mix):
        mixes = _mix3(mix)
        halves = [half_inning_pmf(ph, n, rng, out_mix=mx, baserunning=baserunning)
                  for ph, mx in zip(phases, mixes)]
        seq = []
        for h, reps in zip(halves, phase_innings):
            seq += [h] * reps
        return _convolve_all(seq), halves[-1], mixes[-1]

    a9, a_late, a_mix_l = team_score_pmf(away_phases, away_mix)
    h9, h_late, h_mix_l = team_score_pmf(home_phases, home_mix)

    k = min(len(a9), len(h9))
    a_cdf = np.cumsum(a9)
    p_home_gt = float(np.sum(h9[1:k] * a_cdf[:k - 1]))
    p_tie = float(np.sum(h9[:k] * a9[:k]))

    if ghost_runner:
        a_x = half_inning_pmf(away_phases[-1], n, rng, out_mix=a_mix_l,
                              baserunning=baserunning, start_bases=(0, 1, 0))
        h_x = half_inning_pmf(home_phases[-1], n, rng, out_mix=h_mix_l,
                              baserunning=baserunning, start_bases=(0, 1, 0))
    else:
        a_x, h_x = a_late, h_late
    m = min(len(a_x), len(h_x))
    p_h = float(np.sum(h_x[1:m] * np.cumsum(a_x)[:m - 1]))
    p_a = float(np.sum(a_x[1:m] * np.cumsum(h_x)[:m - 1]))
    p_extra = 0.5 if (p_h + p_a) == 0 else p_h / (p_h + p_a)

    return p_home_gt + p_tie * p_extra


# ----------------------------------------------------------------------------
# 4. data acquisition
# ----------------------------------------------------------------------------
# --- Marcel-style projection helpers ---
MARCEL_WEIGHTS = (5, 4, 3)   # current window, prior season, season before that
HIT_REG_PA = 600.0           # league-average PAs added as regression prior
PIT_REG_BF = 900.0
_CNT_KEYS = ("pa", "bb", "hbp", "h", "d2", "d3", "hr", "k", "go", "ao")


def _stat_to_counts(st: dict) -> dict:
    return {"pa": st.get("plateAppearances") or st.get("battersFaced") or 0,
            "bb": st.get("baseOnBalls", 0), "hbp": st.get("hitByPitch", 0),
            "h": st.get("hits", 0), "d2": st.get("doubles", 0),
            "d3": st.get("triples", 0), "hr": st.get("homeRuns", 0),
            "k": st.get("strikeOuts", 0), "go": st.get("groundOuts", 0),
            "ao": st.get("airOuts", 0)}


def _sum_counts(rows) -> dict:
    tot = {k: 0 for k in _CNT_KEYS}
    for c in rows:
        for k in _CNT_KEYS:
            tot[k] += c[k]
    return tot


def _regressed_rates(c: dict, lg: dict, add_pa: float) -> dict:
    """Rate dict from counts `c` plus `add_pa` league-average PAs of prior."""
    cc = {"pa": c["pa"] + add_pa, "hbp": 0,
          "bb": c["bb"] + c["hbp"] + lg["BB"] * add_pa,
          "h": c["h"] + (lg["1B"] + lg["2B"] + lg["3B"] + lg["HR"]) * add_pa,
          "d2": c["d2"] + lg["2B"] * add_pa, "d3": c["d3"] + lg["3B"] * add_pa,
          "hr": c["hr"] + lg["HR"] * add_pa,
          "k": c["k"], "go": c["go"], "ao": c["ao"]}
    return _rates_from_counts(cc)


def _marcel_blend(sources: list[dict], weights, key: str) -> dict | None:
    acc = {k: 0.0 for k in _CNT_KEYS}
    hit = False
    for w, src in zip(weights, sources):
        c = src.get(key)
        if not c:
            continue
        hit = True
        for k in _CNT_KEYS:
            acc[k] += w * c[k]
    return acc if hit else None


def _team_id_map(season: int) -> dict:
    j = requests.get(f"{API}/teams?sportId=1&season={season}", headers=HEADERS, timeout=30).json()
    return {t["name"]: t["id"] for t in j["teams"]}


def _team_group_splits(group: str, season: int, start: str | None, end: str | None) -> list:
    base = f"{API}/teams/stats"
    if start and end:
        q = (f"?season={season}&stats=byDateRange&startDate={start}&endDate={end}"
             f"&sportIds=1&gameType=R&group={group}")
    else:
        q = f"?season={season}&stats=season&sportIds=1&group={group}"
    return requests.get(base + q, headers=HEADERS, timeout=30).json()["stats"][0]["splits"]


def _era_of(st: dict) -> float:
    er = float(st.get("earnedRuns", 0) or 0)
    ip = float(st.get("inningsPitched", 0) or 0)
    return 9.0 * er / ip if ip else float(st.get("era", 4.3))


def fetch_team_tables(season: int = 2026, start_date: str | None = None,
                      end_date: str | None = None, project: bool = False,
                      prior_seasons=(2025, 2024), weights=MARCEL_WEIGHTS,
                      reg_bf: float = PIT_REG_BF) -> dict:
    """{'bat','pit': {team: rates}, 'lg', 'pit_era': {team: era}, 'lg_era'}.

    project=True -> Marcel-blend each team's hitting & pitching across the current
    window and prior full seasons, regressed toward league.
    """
    hit_cur = {t["team"]["name"]: _stat_to_counts(t["stat"])
               for t in _team_group_splits("hitting", season, start_date, end_date)}
    pit_splits = _team_group_splits("pitching", season, start_date, end_date)
    pit_cur = {t["team"]["name"]: _stat_to_counts(t["stat"]) for t in pit_splits}
    era_cur = {t["team"]["name"]: _era_of(t["stat"]) for t in pit_splits}

    lg_hit = _rates_from_counts({**_sum_counts(hit_cur.values()),
                                 "hbp": _sum_counts(hit_cur.values())["hbp"]})
    lg_pit = _rates_from_counts(_sum_counts(pit_cur.values()))
    lg_era = np.mean(list(era_cur.values()))

    if not project:
        bat = {k: _regressed_rates(c, lg_hit, 0.0) for k, c in hit_cur.items()}
        pit = {k: _regressed_rates(c, lg_pit, 0.0) for k, c in pit_cur.items()}
        return {"bat": bat, "pit": pit, "lg": lg_hit, "lg_pit": lg_pit,
                "pit_era": era_cur, "lg_era": lg_era}

    hit_priors, pit_priors, era_priors = [], [], []
    for y in prior_seasons:
        hs = _team_group_splits("hitting", y, None, None)
        ps = _team_group_splits("pitching", y, None, None)
        hit_priors.append({t["team"]["name"]: _stat_to_counts(t["stat"]) for t in hs})
        pit_priors.append({t["team"]["name"]: _stat_to_counts(t["stat"]) for t in ps})
        era_priors.append({t["team"]["name"]: _era_of(t["stat"]) for t in ps})

    bat, pit, era = {}, {}, {}
    for name in hit_cur:
        b = _marcel_blend([hit_cur, *hit_priors], weights, name)
        bat[name] = _regressed_rates(b or hit_cur[name], lg_hit, reg_bf)
        p = _marcel_blend([pit_cur, *pit_priors], weights, name)
        pit[name] = _regressed_rates(p or pit_cur[name], lg_pit, reg_bf)
        num = den = 0.0
        for w, src in zip(weights, [era_cur, *era_priors]):
            if name in src:
                num += w * src[name]; den += w
        era[name] = (num + lg_era * 1.5) / (den + 1.5) if den else lg_era
    return {"bat": bat, "pit": pit, "lg": lg_hit, "lg_pit": lg_pit,
            "pit_era": era, "lg_era": lg_era}


def fetch_team_bullpen(season: int = 2026, start_date: str | None = None,
                       end_date: str | None = None, project: bool = False,
                       prior_seasons=(2025, 2024), weights=MARCEL_WEIGHTS,
                       reg_bf: float = PIT_REG_BF) -> dict:
    """{team: relief-pitching rates} + '_lg'. Uses the 'rp' situational split."""
    ids = _team_id_map(season)

    def rp_counts(yr, s, e):
        out = {}
        rng_q = f"&startDate={s}&endDate={e}" if s and e else ""
        for name, tid in ids.items():
            try:
                j = requests.get(f"{API}/teams/{tid}/stats?season={yr}&stats=statSplits"
                                 f"&sitCodes=rp&group=pitching&sportId=1&gameType=R{rng_q}",
                                 headers=HEADERS, timeout=30).json()
                out[name] = _stat_to_counts(j["stats"][0]["splits"][0]["stat"])
            except (KeyError, IndexError):
                out[name] = {k: 0 for k in _CNT_KEYS}
        return out

    cur = rp_counts(season, start_date, end_date)
    lg = _rates_from_counts(_sum_counts(cur.values()))
    if not project:
        out = {name: _regressed_rates(c, lg, 0.0) if c["pa"] > 0 else lg
               for name, c in cur.items()}
    else:
        priors = [rp_counts(y, None, None) for y in prior_seasons]
        out = {}
        for name in cur:
            b = _marcel_blend([cur, *priors], weights, name)
            out[name] = _regressed_rates(b or cur[name], lg, reg_bf)
    out["_lg"] = lg
    return out


def fetch_team_game_logs(season: int = 2026) -> dict:
    ids = _team_id_map(season)
    out = {"hit": {}, "pit": {}}
    for name, tid in ids.items():
        for grp, slot in (("hitting", "hit"), ("pitching", "pit")):
            url = (f"{API}/teams/{tid}/stats?stats=gameLog&group={grp}"
                   f"&season={season}&sportId=1&gameType=R")
            splits = requests.get(url, headers=HEADERS, timeout=30).json()["stats"][0]["splits"]
            rows = []
            for s in splits:
                st = s["stat"]
                pa = st.get("plateAppearances") or st.get("battersFaced") or 0
                rows.append({"date": s["date"], "pa": pa,
                             "bb": st.get("baseOnBalls", 0), "hbp": st.get("hitByPitch", 0),
                             "h": st.get("hits", 0), "d2": st.get("doubles", 0),
                             "d3": st.get("triples", 0), "hr": st.get("homeRuns", 0),
                             "k": st.get("strikeOuts", 0), "go": st.get("groundOuts", 0),
                             "ao": st.get("airOuts", 0)})
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("date").reset_index(drop=True)
            out[slot][name] = df
    return out


def _people_hands(pids: list[int]) -> dict:
    """{player_id: 'L'/'R'/'S'} pitch-hand or bat-side, batched."""
    out = {}
    pids = [p for p in pids if p]
    for i in range(0, len(pids), 100):
        chunk = ",".join(str(p) for p in pids[i:i + 100])
        j = requests.get(f"{API}/people?personIds={chunk}", headers=HEADERS, timeout=30).json()
        for p in j.get("people", []):
            hand = (p.get("pitchHand") or p.get("batSide") or {}).get("code")
            if hand:
                out[p["id"]] = hand
    return out


def fetch_probable_starter_eras(start_date: str, end_date: str, season: int = 2026,
                                min_ip: float = 10.0, era_start: str | None = None,
                                era_end: str | None = None, hands: bool = False) -> dict:
    if era_start and era_end:
        url = (f"{API}/stats?stats=byDateRange&group=pitching&season={season}"
               f"&startDate={era_start}&endDate={era_end}"
               f"&sportId=1&gameType=R&playerPool=all&limit=3000")
    else:
        url = (f"{API}/stats?stats=season&group=pitching&season={season}"
               f"&sportId=1&gameType=R&playerPool=all&limit=3000")
    splits = requests.get(url, headers=HEADERS, timeout=60).json()["stats"][0]["splits"]
    era_by_id = {}
    for s in splits:
        try:
            ip = float(s["stat"].get("inningsPitched", 0) or 0)
            era_by_id[s["player"]["id"]] = float(s["stat"]["era"]) if ip >= min_ip else None
        except (KeyError, ValueError, TypeError):
            pass

    sc = requests.get(f"{API}/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
                      f"&hydrate=probablePitcher&gameType=R", headers=HEADERS, timeout=30).json()
    rows, ids = [], set()
    for d in sc["dates"]:
        for g in d["games"]:
            a, h = g["teams"]["away"], g["teams"]["home"]
            ap, hp = a.get("probablePitcher"), h.get("probablePitcher")
            rows.append((g["gameDate"][:10], a["team"]["name"], h["team"]["name"], ap, hp))
            ids.update(x["id"] for x in (ap, hp) if x)
    hand_by_id = _people_hands(list(ids)) if hands else {}

    out = {}
    for gd, away, home, ap, hp in rows:
        out[(gd, away, home)] = {
            "away_era": era_by_id.get(ap["id"]) if ap else None,
            "home_era": era_by_id.get(hp["id"]) if hp else None,
            "away_name": ap["fullName"] if ap else None,
            "home_name": hp["fullName"] if hp else None,
            "away_hand": hand_by_id.get(ap["id"]) if ap else None,
            "home_hand": hand_by_id.get(hp["id"]) if hp else None,
        }
    return out


def _pitcher_counts(season: int, start: str | None = None, end: str | None = None) -> dict:
    if start and end:
        url = (f"{API}/stats?stats=byDateRange&group=pitching&season={season}"
               f"&startDate={start}&endDate={end}&sportId=1&gameType=R&playerPool=all&limit=4000")
    else:
        url = (f"{API}/stats?stats=season&group=pitching&season={season}"
               f"&sportId=1&gameType=R&playerPool=all&limit=4000")
    splits = requests.get(url, headers=HEADERS, timeout=60).json()["stats"][0]["splits"]
    out, tot = {}, {k: 0 for k in _CNT_KEYS}
    for s in splits:
        c = _stat_to_counts(s["stat"])           # 'pa' <- battersFaced via fallback
        if c["pa"] < 1:
            continue
        out[s["player"]["id"]] = c
        for k in _CNT_KEYS:
            tot[k] += c[k]
    out["_lg"] = tot
    return out


def fetch_starter_rates(start_date: str, end_date: str, season: int = 2026, *,
                        rate_start: str | None = None, rate_end: str | None = None,
                        project: bool = False, prior_seasons=(2025, 2024),
                        weights=MARCEL_WEIGHTS, reg_bf: float = 300.0,
                        min_bf: int = 40) -> dict:
    """{(game_date, away_team, home_team): {'away','home': per-BF rate dict|None,
        'away_hand','home_hand','away_name','home_name'}}.

    Each probable starter's OWN component rates (1B/2B/3B/HR/BB + K,GO,AO mix,
    per batter faced), regressed toward league with `reg_bf` batters of prior.
    project=True Marcel-blends `prior_seasons` in first. Starters with < `min_bf`
    batters faced (and no prior data when projecting) come back None -> the
    caller falls back to team pitching + the ERA scale.
    """
    cur = _pitcher_counts(season, rate_start, rate_end)
    lg = _rates_from_counts(cur["_lg"])
    priors = [_pitcher_counts(y) for y in prior_seasons] if project else []

    def rate_for(pid):
        if pid is None:
            return None
        if project:
            b = _marcel_blend([cur, *priors], weights, pid)
            if b is None or (b["pa"] < min_bf and cur.get(pid, {}).get("pa", 0) < min_bf):
                return None
            return _regressed_rates(b, lg, reg_bf)
        c = cur.get(pid)
        if not c or c["pa"] < min_bf:
            return None
        return _regressed_rates(c, lg, reg_bf)

    sc = requests.get(f"{API}/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
                      f"&hydrate=probablePitcher&gameType=R", headers=HEADERS, timeout=30).json()
    rows, ids = [], set()
    for d in sc["dates"]:
        for g in d["games"]:
            a, h = g["teams"]["away"], g["teams"]["home"]
            ap, hp = a.get("probablePitcher"), h.get("probablePitcher")
            rows.append((g["gameDate"][:10], a["team"]["name"], h["team"]["name"], ap, hp))
            ids.update(x["id"] for x in (ap, hp) if x)
    hand_by_id = _people_hands(list(ids))

    out = {}
    for gd, away, home, ap, hp in rows:
        out[(gd, away, home)] = {
            "away": rate_for(ap["id"] if ap else None),
            "home": rate_for(hp["id"] if hp else None),
            "away_hand": hand_by_id.get(ap["id"]) if ap else None,
            "home_hand": hand_by_id.get(hp["id"]) if hp else None,
            "away_name": ap["fullName"] if ap else None,
            "home_name": hp["fullName"] if hp else None,
        }
    return out


def fetch_team_platoon(season: int = 2026, start_date: str | None = None,
                       end_date: str | None = None) -> dict:
    """{team: {'L': mult, 'R': mult}} -- offense reach multiplier vs LHP / RHP,
    normalised to the team's overall OPS. One call per team."""
    ids = _team_id_map(season)
    if start_date and end_date:
        rng_q = f"&startDate={start_date}&endDate={end_date}"
        base_stat = "byDateRange"
    else:
        rng_q, base_stat = "", "season"
    out = {}
    for name, tid in ids.items():
        try:
            j = requests.get(f"{API}/teams/{tid}/stats?season={season}&stats=statSplits"
                             f"&sitCodes=vl,vr&group=hitting&sportId=1&gameType=R{rng_q}",
                             headers=HEADERS, timeout=30).json()
            ov = requests.get(f"{API}/teams/{tid}/stats?season={season}&stats={base_stat}"
                              f"&group=hitting&sportId=1&gameType=R{rng_q}",
                              headers=HEADERS, timeout=30).json()
            base_ops = float(ov["stats"][0]["splits"][0]["stat"]["ops"])
            d = {"L": 1.0, "R": 1.0}
            for sp in j["stats"][0]["splits"]:
                code = sp["split"]["code"]
                ops = float(sp["stat"]["ops"])
                if base_ops > 0:
                    d["L" if code == "vl" else "R"] = ops / base_ops
            out[name] = d
        except (KeyError, IndexError, ValueError):
            out[name] = {"L": 1.0, "R": 1.0}
    return out


def fetch_results(start_date: str, end_date: str) -> pd.DataFrame:
    sc = requests.get(f"{API}/schedule?sportId=1&startDate={start_date}"
                      f"&endDate={end_date}&gameType=R", headers=HEADERS, timeout=30).json()
    rows = []
    for d in sc["dates"]:
        for g in d["games"]:
            if g.get("status", {}).get("statusCode") != "F":
                continue
            a, h = g["teams"]["away"], g["teams"]["home"]
            if "score" not in a or "score" not in h:
                continue
            rows.append({"game_date": g["gameDate"][:10],
                         "away_team": a["team"]["name"], "home_team": h["team"]["name"],
                         "away_score": a["score"], "home_score": h["score"]})
    df = pd.DataFrame(rows)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    return df[df["home_score"] != df["away_score"]].reset_index(drop=True)


def fetch_lineups(start_date: str, end_date: str) -> dict:
    sc = requests.get(f"{API}/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
                      f"&hydrate=lineups&gameType=R", headers=HEADERS, timeout=30).json()
    out = {}
    for d in sc["dates"]:
        for g in d["games"]:
            lu = g.get("lineups", {})
            ap = [p["id"] for p in lu.get("awayPlayers", [])]
            hp = [p["id"] for p in lu.get("homePlayers", [])]
            if len(ap) == 9 and len(hp) == 9:
                key = (g["gameDate"][:10], g["teams"]["away"]["team"]["name"],
                       g["teams"]["home"]["team"]["name"])
                out[key] = {"away": ap, "home": hp}
    return out


def fetch_starter_rest(start_date: str, end_date: str, season: int = 2026) -> dict:
    sc = requests.get(f"{API}/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
                      f"&hydrate=probablePitcher&gameType=R", headers=HEADERS, timeout=30).json()
    games, starter_ids = [], set()
    for d in sc["dates"]:
        for g in d["games"]:
            a, h = g["teams"]["away"], g["teams"]["home"]
            ap, hp = a.get("probablePitcher"), h.get("probablePitcher")
            aid = ap["id"] if ap else None
            hid = hp["id"] if hp else None
            games.append((g["gameDate"][:10], a["team"]["name"], h["team"]["name"], aid, hid))
            starter_ids.update(x for x in (aid, hid) if x)

    appearances = {}
    for pid in starter_ids:
        try:
            sp = requests.get(f"{API}/people/{pid}/stats?stats=gameLog&group=pitching"
                              f"&season={season}", headers=HEADERS, timeout=30
                              ).json()["stats"][0]["splits"]
            appearances[pid] = sorted(s["date"] for s in sp)
        except (IndexError, KeyError):
            appearances[pid] = []

    def rest_for(pid, gd):
        if not pid:
            return None
        prior = [x for x in appearances.get(pid, []) if x < gd]
        if not prior:
            return None
        return int((pd.Timestamp(gd) - pd.Timestamp(prior[-1])).days)

    return {(gd, aw, hm): {"away_rest": rest_for(aid, gd), "home_rest": rest_for(hid, gd)}
            for gd, aw, hm, aid, hid in games}


def _player_hit_counts(season: int, start: str | None = None, end: str | None = None) -> dict:
    if start and end:
        url = (f"{API}/stats?stats=byDateRange&group=hitting&season={season}"
               f"&startDate={start}&endDate={end}&sportId=1&gameType=R&playerPool=all&limit=6000")
    else:
        url = (f"{API}/stats?stats=season&group=hitting&season={season}"
               f"&sportId=1&gameType=R&playerPool=all&limit=6000")
    splits = requests.get(url, headers=HEADERS, timeout=60).json()["stats"][0]["splits"]
    out, tot = {}, {k: 0 for k in _CNT_KEYS}
    for s in splits:
        c = _stat_to_counts(s["stat"])
        if c["pa"] < 1:
            continue
        out[s["player"]["id"]] = c
        for k in _CNT_KEYS:
            tot[k] += c[k]
    out["_lg"] = tot
    return out


def fetch_player_hit_rates(season: int = 2026, shrink_pa: float = 100.0,
                           start_date: str | None = None, end_date: str | None = None,
                           project: bool = False, prior_seasons=(2025, 2024),
                           weights=MARCEL_WEIGHTS, reg_pa: float = HIT_REG_PA) -> dict:
    """{player_id: per-PA rates} + '_league'.

    project=False -> season-to-date rate regressed toward league with `shrink_pa`.
    project=True  -> Marcel: weighted blend of the current window and each prior
                     full season, then `reg_pa` league PAs of regression.
    """
    cur = _player_hit_counts(season, start_date, end_date)
    lg = _rates_from_counts({**cur["_lg"], "hbp": cur["_lg"]["hbp"]})

    if not project:
        return {"_league": lg,
                **{pid: _regressed_rates(c, lg, shrink_pa)
                   for pid, c in cur.items() if pid != "_lg"}}

    priors = [_player_hit_counts(y) for y in prior_seasons]
    ids = set().union(*[set(s) for s in [cur, *priors]]) - {"_lg"}
    out = {"_league": lg}
    for pid in ids:
        blend = _marcel_blend([cur, *priors], weights, pid)
        if blend is None:
            continue
        out[pid] = _regressed_rates(blend, lg, reg_pa)
    return out


# ----------------------------------------------------------------------------
# 5. rolling rates
# ----------------------------------------------------------------------------
def rolling_rates(log_df: pd.DataFrame, as_of: str, window: int, fallback: dict,
                  min_games: int = 3, shrink_pa: float = 0.0) -> dict:
    if log_df is None or log_df.empty:
        return fallback
    prior = log_df[log_df["date"] < as_of]
    if len(prior) < min_games:
        return fallback
    cols = [c for c in _COUNT_COLS + ["k", "go", "ao"] if c in prior.columns]
    c = prior.tail(window)[cols].sum().astype(float).to_dict()
    if c.get("pa", 0) <= 0:
        return fallback
    if shrink_pa > 0:
        c["pa"] += shrink_pa
        for col, ev in (("bb", "BB"), ("d2", "2B"), ("d3", "3B"), ("hr", "HR")):
            c[col] = c.get(col, 0) + fallback[ev] * shrink_pa
        c["h"] = c.get("h", 0) + (fallback["1B"] + fallback["2B"] + fallback["3B"]
                                  + fallback["HR"]) * shrink_pa
    c.setdefault("hbp", 0)
    r = _rates_from_counts(c)
    if "k" not in c:
        r["_mix"] = fallback.get("_mix", (.33, .34, .33))
    return r


# ----------------------------------------------------------------------------
# 6. predict every game
# ----------------------------------------------------------------------------
def predict_games(games: pd.DataFrame, tables: dict, *, logs: dict | None = None,
                  window: int | None = None, shrink_pa: float = 0.0,
                  lineups: dict | None = None, player_rates: dict | None = None,
                  starters: dict | None = None, starter_strength: float = 0.3,
                  starter_rates: dict | None = None,
                  rest: dict | None = None, rest_strength: float = 1.0,
                  home_field: float = 0.024, platoon: dict | None = None,
                  bullpen_rates: dict | None = None,
                  realism: bool = True, park: bool | None = None,
                  baserunning: bool | None = None, bullpen: bool | None = None,
                  ghost_runner: bool | None = None, roe: float = 0.012,
                  tto: float = 0.03, bullpen_factor: float = 0.96,
                  n: int = 40_000, seed: int = 7,
                  date_col: str = "game_date") -> pd.Series:
    """P(home win) per row. `realism` toggles gaps 1-4 as a bundle; individual
    flags override it. `bullpen_rates` (fetch_team_bullpen) makes innings 7-9 a
    real matchup vs the opposing pen. `starter_rates` (fetch_starter_rates) makes
    innings 1-6 a real matchup vs the probable starter's own component rates,
    replacing the ERA scale for games where those rates are available."""
    park = realism if park is None else park
    baserunning = realism if baserunning is None else baserunning
    bullpen = realism if bullpen is None else bullpen
    ghost_runner = realism if ghost_runner is None else ghost_runner

    lg = tables["lg"]
    use_lineups = lineups is not None and player_rates is not None

    def team_rates(kind, team, as_of):
        season = tables["bat" if kind == "hit" else "pit"][team]
        if window and logs is not None:
            return rolling_rates(logs[kind][team], as_of, window, season, shrink_pa=shrink_pa)
        return season

    def prep(v6, pf):
        return add_roe(apply_park(v6, pf), roe) if park else add_roe(v6, roe)

    out = np.empty(len(games), dtype=float)
    cache: dict[tuple, float] = {}
    it = zip(games["away_team"], games["home_team"], games[date_col].astype(str))
    for i, (away, home, gdate) in enumerate(it):
        gkey = (gdate, away, home)
        g = starters.get(gkey, {}) if starters is not None else {}
        sr = starter_rates.get(gkey, {}) if starter_rates is not None else {}
        away_sp = sr.get("away")            # away starter's own component rates
        home_sp = sr.get("home")

        hs = as_ = 1.0                      # ERA + rest scale (fallback path only)
        if starters is not None:
            hs = starter_scale(g.get("home_era"), tables["pit_era"].get(home), starter_strength)
            as_ = starter_scale(g.get("away_era"), tables["pit_era"].get(away), starter_strength)
        if rest is not None:
            r = rest.get(gkey, {})
            hs *= rest_scale(r.get("home_rest"), rest_strength)
            as_ *= rest_scale(r.get("away_rest"), rest_strength)

        home_hand = sr.get("home_hand") or g.get("home_hand")
        away_hand = sr.get("away_hand") or g.get("away_hand")
        home_hf = 1.0 + home_field
        away_hf = 1.0 - home_field
        if platoon is not None:
            if home_hand in ("L", "R"):
                away_hf *= platoon.get(away, {}).get(home_hand, 1.0)
            if away_hand in ("L", "R"):
                home_hf *= platoon.get(home, {}).get(away_hand, 1.0)

        lu = lineups.get(gkey) if use_lineups else None
        lu_tag = (tuple(lu["away"]), tuple(lu["home"])) if lu else None
        pf = PARK_FACTORS.get(home, (1.0, 1.0)) if park else (1.0, 1.0)

        key = (away, home, gdate, window, lu_tag,
               round(away_hf, 3), round(home_hf, 3), round(hs, 3), round(as_, 3),
               pf, realism, baserunning, bullpen, ghost_runner,
               bullpen_rates is not None, away_sp is not None, home_sp is not None)
        if key not in cache:
            away_pit = team_rates("pit", away, gdate)
            home_pit = team_rates("pit", home, gdate)
            if lu:
                away_bat = lineup_offense_rates(lu["away"], player_rates,
                                                team_rates("hit", away, gdate)["_mix"])
                home_bat = lineup_offense_rates(lu["home"], player_rates,
                                                team_rates("hit", home, gdate)["_mix"])
            else:
                away_bat = team_rates("hit", away, gdate)
                home_bat = team_rates("hit", home, gdate)

            def phase_vecs(bat, opp_pit, opp_team, opp_sp, hf, era_sc):
                # innings 1-6: the starter's own rates if we have them, else team + ERA
                if opp_sp is not None:
                    p12, s12 = opp_sp, hf
                else:
                    p12, s12 = opp_pit, hf * era_sc
                early = prep(matchup_probs(bat, p12, lg, s12), pf)
                mix12 = blend_out_mix(bat, p12)
                if not bullpen:
                    return [early, early, early], [mix12, mix12, mix12]
                mid = prep(matchup_probs(bat, p12, lg, s12 * (1.0 + tto)), pf)
                if bullpen_rates and opp_team in bullpen_rates:
                    pen = bullpen_rates[opp_team]
                    late = prep(matchup_probs(bat, pen, lg, hf), pf)
                    mixL = blend_out_mix(bat, pen)
                else:
                    late, mixL = scale_reach(early, bullpen_factor), mix12
                return [early, mid, late], [mix12, mix12, mixL]

            # away bats vs the HOME starter; home bats vs the AWAY starter
            away_phases, away_mix = phase_vecs(away_bat, home_pit, home, home_sp, away_hf, hs)
            home_phases, home_mix = phase_vecs(home_bat, away_pit, away, away_sp, home_hf, as_)

            rng = np.random.default_rng(seed + zlib.crc32(repr(key).encode()))
            cache[key] = home_win_probability(
                away_phases, home_phases, away_mix=away_mix, home_mix=home_mix,
                baserunning=baserunning, ghost_runner=ghost_runner, n=n, rng=rng)
        out[i] = cache[key]

    return pd.Series(out, index=games.index, name="mc_home_win_prob")
