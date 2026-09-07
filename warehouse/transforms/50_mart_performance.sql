-- Scoring metrics per (source): Brier, log loss, accuracy, ECE, and the flat
-- betting P&L / ROI. This is the one query the dashboard and the resume both quote.
CREATE OR REPLACE TABLE mart_performance AS
WITH preds AS (
    SELECT source, p_blend AS p, home_win FROM stg_predictions WHERE home_win IS NOT NULL
    UNION ALL
    SELECT 'market', mkt_home_prob, home_win
    FROM stg_predictions
    WHERE source = 'walkforward' AND home_win IS NOT NULL AND mkt_home_prob IS NOT NULL
),
scores AS (
    SELECT source,
        COUNT(*)                                        AS n_games,
        ROUND(AVG((p - home_win) * (p - home_win)), 4)  AS brier,
        ROUND(AVG(-(home_win * LN(GREATEST(p, 1e-9))
                 + (1 - home_win) * LN(GREATEST(1 - p, 1e-9)))), 4) AS log_loss,
        ROUND(AVG(((p >= 0.5)::INT = home_win)::INT), 4) AS accuracy
    FROM preds GROUP BY source
),
ece AS (
    SELECT source, ROUND(SUM(n * gap) / SUM(n), 4) AS ece
    FROM mart_calibration GROUP BY source
),
betting AS (
    SELECT source,
        COUNT(*)                       AS n_bets,
        ROUND(AVG(won), 3)             AS win_rate,
        ROUND(SUM(pnl), 2)             AS net_pnl,
        ROUND(SUM(pnl) / NULLIF(SUM(stake), 0), 4) AS roi
    FROM mart_bets GROUP BY source
)
SELECT s.source, s.n_games, s.brier, s.log_loss, s.accuracy, e.ece,
       b.n_bets, b.win_rate, b.net_pnl, b.roi
FROM scores s
LEFT JOIN ece e USING (source)
LEFT JOIN betting b USING (source)
ORDER BY s.brier;
