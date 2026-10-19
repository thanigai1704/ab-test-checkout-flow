-- ============================================================================
--  A/B TEST CHECKOUT FLOW — DATA EXTRACTION & CLEANING (SQL)
--  Project: Designing and Analyzing an A/B Test for a Checkout Flow
--  Dialect: PostgreSQL (notes given for MySQL / BigQuery / Snowflake)
--
--  Source of record: the Udacity e-commerce A/B experiment export
--  (ab_data.csv, 294,478 rows, user-level log of a real Jan-2017 test)
--  plus the country dimension table (countries.csv, 290,584 rows).
--
--  In production this file would live in the warehouse (Snowflake /
--  BigQuery) and be orchestrated by Airflow / dbt. Here it documents the
--  exact relational logic that the pandas pipeline in ab_stats.py and
--  AB_Test_Checkout_Flow_Analysis.ipynb reproduces 1:1.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- STEP 0 — LANDING / STAGING
-- Raw export lands in a staging schema untouched (audit trail).
--   rows_in: 294,478
-- ----------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS staging.experiment_raw (
    user_id       BIGINT,
    "timestamp"   TIMESTAMP,          -- real timestamps: 2017-01-02 .. 2017-01-24
    group_name    TEXT,               -- 'control' | 'treatment'
    landing_page  TEXT,               -- 'old_page' | 'new_page'
    converted     SMALLINT            -- 0/1
);

\copy staging.experiment_raw FROM 'data/ab_data.csv' WITH (FORMAT csv, HEADER true);

-- ----------------------------------------------------------------------------
-- STEP 1 — DATA QUALITY AUDIT (run BEFORE deleting anything)
-- Quantifies every defect so the cleaning is defensible in a review.
-- ----------------------------------------------------------------------------

-- 1a. Assignment-integrity violations: group and landing_page disagree.
--     These users saw the OPPOSITE page of their assigned arm — the exposure
--     log contradicts the assignment log. Counted: 3,893 rows.
SELECT COUNT(*) AS mismatched_assignment_rows
FROM staging.experiment_raw
WHERE (group_name = 'treatment' AND landing_page = 'old_page')
   OR (group_name = 'control'   AND landing_page = 'new_page');

-- 1b. Duplicate users (a user appearing more than once could be counted in
--     both arms or double-counted in one). Counted: 3,894 user_ids.
SELECT COUNT(*) AS duplicate_user_ids
FROM (
    SELECT user_id
    FROM staging.experiment_raw
    GROUP BY user_id
    HAVING COUNT(*) > 1
) d;

-- 1c. Sanity checks on value domains and the metric itself.
SELECT
    COUNT(*)                                                          AS total_rows,
    COUNT(DISTINCT user_id)                                           AS distinct_users,
    COUNT(*) - COUNT(user_id)                                         AS null_user_ids,
    COUNT(*) FILTER (WHERE group_name NOT IN ('control','treatment')) AS bad_group_values,
    COUNT(*) FILTER (WHERE converted NOT IN (0,1))                    AS bad_converted_values,
    MIN("timestamp")                                                  AS first_exposure,
    MAX("timestamp")                                                  AS last_exposure
FROM staging.experiment_raw;

-- ----------------------------------------------------------------------------
-- STEP 2 — CLEANING (the C in ETL)
-- Rule 1: drop rows where exposure contradicts assignment.
-- Rule 2: keep exactly ONE row per user (deterministic tie-break: latest
--         timestamp, then the raw row order via ctid as a final tiebreak).
-- Result: 294,478 -> 290,584 analytic rows (matches the pandas pipeline).
-- ----------------------------------------------------------------------------
CREATE TABLE analytics.experiment_clean AS
WITH dedup AS (
    SELECT *
    FROM (
        SELECT
            e.*,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY "timestamp" DESC, ctid DESC   -- MySQL: ORDER BY `timestamp` DESC only
            ) AS rn
        FROM staging.experiment_raw e
        WHERE NOT ( (group_name = 'treatment' AND landing_page = 'old_page')
                 OR (group_name = 'control'   AND landing_page = 'new_page') )
    ) t
    WHERE rn = 1
)
SELECT
    user_id,
    "timestamp",
    group_name                              AS experiment_group,
    landing_page,
    converted::BOOLEAN                      AS converted,
    DATE("timestamp")                       AS exposure_date
FROM dedup;

ANALYZE analytics.experiment_clean;

-- Post-clean assertion (should return 0 / 0 / 290,584):
SELECT
    COUNT(*) FILTER (WHERE (experiment_group='treatment' AND landing_page='old_page')
                        OR (experiment_group='control'  AND landing_page='new_page')) AS residual_mismatches,
    COUNT(*) - COUNT(DISTINCT user_id)                                                            AS residual_duplicates,
    COUNT(*)                                                                                      AS clean_rows
FROM analytics.experiment_clean;

-- ----------------------------------------------------------------------------
-- STEP 3 — SRM (SAMPLE RATIO MISMATCH) VALIDATION
-- Before measuring ANY effect, verify the splitter actually produced ~50/50.
-- chi-square with 1 df, alpha = 0.001 (deliberately strict).
-- Observed: control 145,274 / treatment 145,310 -> ratio 1.0002, p = 0.947 -> PASS
-- ----------------------------------------------------------------------------
SELECT
    experiment_group,
    COUNT(*)                                                AS users,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)      AS pct_share
FROM analytics.experiment_clean
GROUP BY experiment_group
ORDER BY experiment_group;

-- ----------------------------------------------------------------------------
-- STEP 4 — METRIC AGGREGATION (what the dashboard / BI layer consumes)
-- These queries produce the same numbers as exports/*.csv.
-- ----------------------------------------------------------------------------

-- 4a. Arm-level conversion (the headline metric).
--     control: 12.0386%   treatment: 11.8808%   diff: -0.16pp
SELECT
    experiment_group,
    COUNT(*)                                        AS users,
    SUM(converted::INT)                             AS conversions,
    ROUND(AVG(converted::INT) * 100, 4)             AS conversion_rate_pct
FROM analytics.experiment_clean
GROUP BY experiment_group;

-- 4b. Daily trend (feeds the peeking / novelty audit visuals).
SELECT
    exposure_date,
    experiment_group,
    COUNT(*)                            AS users,
    ROUND(AVG(converted::INT) * 100, 4) AS conv_pct
FROM analytics.experiment_clean
GROUP BY exposure_date, experiment_group
ORDER BY exposure_date, experiment_group;

-- 4c. Country segments (Simpson's paradox audit; join to dimension table).
--     UK +0.11pp | US -0.22pp | CA -0.69pp — no segment reverses the verdict.
CREATE TABLE IF NOT EXISTS analytics.countries (
    user_id BIGINT PRIMARY KEY,
    country TEXT                                -- 'US' | 'UK' | 'CA'
);
-- \copy analytics.countries FROM 'data/countries.csv' WITH (FORMAT csv, HEADER true);

SELECT
    c.country,
    e.experiment_group,
    COUNT(*)                            AS users,
    ROUND(AVG(e.converted::INT) * 100, 4) AS conv_pct
FROM analytics.experiment_clean e
JOIN analytics.countries c USING (user_id)
GROUP BY c.country, e.experiment_group
ORDER BY c.country, e.experiment_group;

-- ----------------------------------------------------------------------------
-- STEP 5 — SECOND EXPERIMENT (Cookie Cats): identical relational discipline
-- Gate placement test on 90,189 real players. Metric: 7-day retention.
-- gate_30 19.02% vs gate_40 18.20%, two-sided p = 0.0016 -> keep gate_30.
-- ----------------------------------------------------------------------------
SELECT
    version,
    COUNT(*)                                          AS players,
    ROUND(AVG(retention_1::INT) * 100, 2)             AS d1_retention_pct,
    ROUND(AVG(retention_7::INT) * 100, 2)             AS d7_retention_pct
FROM staging.cookie_cats_raw           -- loaded the same way as STEP 0
GROUP BY version;
