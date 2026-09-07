-- All model predictions on one grain: (game_id, source).
--   source = 'walkforward'  -> the offline expanding-window backtest
--   source = 'forward'      -> the live daily loop
-- p_blend is the deployed combiner: 0.40 * season-to-date + 0.60 * projection.
CREATE OR REPLACE TABLE stg_predictions AS
WITH wf AS (
    SELECT
        strftime(game_date::DATE, '%Y-%m-%d') || '|' || away_team || '|' || home_team AS game_id,
        game_date::DATE                     AS game_date,
        'walkforward'                       AS source,
        CAST(p_std  AS DOUBLE)              AS p_std,
        CAST(p_proj AS DOUBLE)              AS p_proj,
        0.40 * CAST(p_std AS DOUBLE) + 0.60 * CAST(p_proj AS DOUBLE) AS p_blend,
        CAST(market AS DOUBLE)              AS mkt_home_prob,
        CAST(home_win AS INT)              AS home_win
    FROM raw_wf_predictions
),
fw AS (
    SELECT
        strftime(game_date::DATE, '%Y-%m-%d') || '|' || away_team || '|' || home_team AS game_id,
        game_date::DATE                     AS game_date,
        'forward'                          AS source,
        NULL::DOUBLE                        AS p_std,
        NULL::DOUBLE                        AS p_proj,
        TRY_CAST(model_p_home AS DOUBLE)    AS p_blend,
        TRY_CAST(entry_p_home AS DOUBLE)    AS mkt_home_prob,
        TRY_CAST(home_win AS DOUBLE)::INT   AS home_win
    FROM raw_forward_log
    WHERE status = 'settled' AND TRY_CAST(model_p_home AS DOUBLE) IS NOT NULL
)
SELECT * FROM wf
UNION ALL
SELECT * FROM fw;
