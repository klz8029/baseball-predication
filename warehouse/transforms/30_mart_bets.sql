-- Simulate the flat-stake betting rule in SQL: bet the side the model favours by
-- more than `edge_threshold`, skip extreme prices. Works for both sources; the
-- forward rows reflect bets that were actually logged.
CREATE OR REPLACE TABLE mart_bets AS
WITH params AS (SELECT 0.03 AS edge_threshold, 10 AS contracts, 0.07 AS fee_rate),
scored AS (
    SELECT
        p.game_id, p.game_date, p.source,
        p.p_blend, p.mkt_home_prob, p.home_win,
        p.p_blend - p.mkt_home_prob                         AS edge_home,
        CASE WHEN p.p_blend >= p.mkt_home_prob THEN 'home' ELSE 'away' END AS side,
        CASE WHEN p.p_blend >= p.mkt_home_prob
             THEN p.mkt_home_prob ELSE 1 - p.mkt_home_prob END            AS price,
        CASE WHEN p.p_blend >= p.mkt_home_prob
             THEN p.p_blend ELSE 1 - p.p_blend END                        AS model_prob
    FROM stg_predictions p
    WHERE p.home_win IS NOT NULL AND p.mkt_home_prob IS NOT NULL
),
bets AS (
    SELECT s.*,
        (SELECT contracts FROM params)                     AS contracts,
        CASE WHEN s.side = 'home' THEN s.home_win = 1 ELSE s.home_win = 0 END AS won
    FROM scored s, params
    WHERE ABS(s.edge_home) > params.edge_threshold
      AND s.price BETWEEN 0.05 AND 0.95
)
SELECT
    b.game_id, b.source, b.side,
    ROUND(b.price, 3)        AS price,
    ROUND(b.model_prob, 3)   AS model_prob,
    ROUND(ABS(b.edge_home), 3) AS edge,
    b.contracts,
    b.won::INT               AS won,
    -- flat P&L per bet, minus Kalshi's ceil(0.07 * n * p * (1-p)) fee
    ROUND(
        b.contracts * (CASE WHEN b.won THEN 1.0 - b.price ELSE -b.price END)
        - CEIL((SELECT fee_rate FROM params) * b.contracts * b.price * (1 - b.price) * 100) / 100
    , 3)                     AS pnl,
    ROUND(b.contracts * b.price, 3) AS stake,
    b.game_date
FROM bets b;
