-- One clean row per game, from the monthly odds+results CSVs.
-- game_id is the natural key used everywhere downstream.
CREATE OR REPLACE TABLE stg_games AS
WITH u AS (
    SELECT * FROM raw_odds
),
d AS (
    SELECT
        game_date::DATE                              AS game_date,
        away_team, home_team,
        CAST(away_score AS INT)                      AS away_score,
        CAST(home_score AS INT)                      AS home_score,
        CAST(away_implied_win_prob AS DOUBLE)        AS mkt_away_prob,
        CAST(home_implied_win_prob AS DOUBLE)        AS mkt_home_prob,
        ROW_NUMBER() OVER (
            PARTITION BY game_date, away_team, home_team
            ORDER BY start_time_utc DESC
        ) AS rn
    FROM u
    WHERE home_score <> away_score          -- drop the rare tie / suspended
)
SELECT
    strftime(game_date, '%Y-%m-%d') || '|' || away_team || '|' || home_team AS game_id,
    game_date, away_team, home_team, away_score, home_score,
    (home_score > away_score)::INT       AS home_win,
    mkt_home_prob, mkt_away_prob,
    -- month bucket for the walk-forward view
    strftime(game_date, '%Y-%m')         AS month
FROM d
WHERE rn = 1;
