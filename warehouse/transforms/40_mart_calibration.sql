-- Reliability table: bucket P(home win) into deciles, compare predicted to actual.
-- A calibrated model has mean_pred ~ actual_rate in every bin.
CREATE OR REPLACE TABLE mart_calibration AS
WITH base AS (
    SELECT source,
           LEAST(9, FLOOR(p_blend * 10))::INT AS bin,
           p_blend, home_win
    FROM stg_predictions
    WHERE home_win IS NOT NULL
    UNION ALL
    SELECT 'market' AS source,
           LEAST(9, FLOOR(mkt_home_prob * 10))::INT AS bin,
           mkt_home_prob AS p_blend, home_win
    FROM stg_predictions
    WHERE home_win IS NOT NULL AND mkt_home_prob IS NOT NULL AND source = 'walkforward'
)
SELECT
    source,
    bin,
    (bin / 10.0)                       AS bin_lo,
    (bin / 10.0) + 0.1                 AS bin_hi,
    COUNT(*)                           AS n,
    ROUND(AVG(p_blend), 4)             AS mean_pred,
    ROUND(AVG(home_win), 4)            AS actual_rate,
    ROUND(ABS(AVG(p_blend) - AVG(home_win)), 4) AS gap
FROM base
GROUP BY source, bin
ORDER BY source, bin;
