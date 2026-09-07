-- Live betting P&L over time: daily totals plus cumulative P&L, a 20-bet rolling
-- ROI, and rolling closing-line-value from the forward log. Window functions.
CREATE OR REPLACE TABLE mart_daily AS
WITH fw AS (
    SELECT
        game_date::DATE AS game_date,
        TRY_CAST(pnl_1c AS DOUBLE)  AS pnl,
        CASE WHEN bet_side = 'home' THEN TRY_CAST(entry_p_home AS DOUBLE)
             ELSE 1 - TRY_CAST(entry_p_home AS DOUBLE) END AS stake,
        TRY_CAST(clv AS DOUBLE) AS clv,
        (TRY_CAST(pnl_1c AS DOUBLE) > 0)::INT AS won
    FROM raw_forward_log
    WHERE status = 'settled' AND bet_side IN ('home', 'away')
      AND TRY_CAST(pnl_1c AS DOUBLE) IS NOT NULL
),
by_day AS (
    SELECT game_date,
           COUNT(*)        AS bets,
           SUM(won)        AS wins,
           ROUND(SUM(stake), 2) AS staked,
           ROUND(SUM(pnl), 2)   AS pnl,
           ROUND(AVG(clv), 4)   AS clv_mean
    FROM fw GROUP BY game_date
),
seq AS (  -- per-bet ordering for the rolling windows
    SELECT *, ROW_NUMBER() OVER (ORDER BY game_date) AS k FROM fw
)
SELECT
    d.*,
    ROUND(SUM(d.pnl) OVER (ORDER BY d.game_date), 2)                       AS cum_pnl,
    ROUND(SUM(d.bets) OVER (ORDER BY d.game_date), 0)                      AS cum_bets,
    (SELECT ROUND(SUM(s.pnl) / NULLIF(SUM(s.stake), 0), 4)
       FROM seq s WHERE s.game_date <= d.game_date
        AND s.k > (SELECT MAX(k) FROM seq WHERE game_date <= d.game_date) - 20) AS roi_last20,
    (SELECT ROUND(AVG((s.clv > 0)::INT), 3)
       FROM seq s WHERE s.game_date <= d.game_date AND s.clv IS NOT NULL
        AND s.k > (SELECT MAX(k) FROM seq WHERE game_date <= d.game_date) - 30) AS beat_close_last30
FROM by_day d
ORDER BY d.game_date;
