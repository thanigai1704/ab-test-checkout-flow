# 🧪 Designing and Analyzing an A/B Test for a Checkout Flow

**Statistically rigorous A/B testing end-to-end on 100% REAL data: markdown-first notebook covering power analysis → data hygiene on real defects (mismatched assignments, duplicate users) → SRM checks → two-proportion Z-tests (+ chi-square twin cross-check) → confidence intervals & effect sizes → real-timestamp peeking audit → novelty & Simpson's paradox checks — across two real experiments: an e-commerce page test (290,584 clean users) and Cookie Cats (90,189 players).**

**Live demo (Streamlit):** *run locally with `streamlit run app.py` (deploy URL lands here after Cloud push)*
**Report:** [`report.md`](report.md) · **Notebook:** [`AB_Test_Checkout_Flow_Analysis.ipynb`](AB_Test_Checkout_Flow_Analysis.ipynb) · **Power BI guide:** [`POWERBI_BUILD_GUIDE.md`](POWERBI_BUILD_GUIDE.md)

---

## Business Problem

> An e-commerce company redesigned the page sitting at the top of the purchase funnel. Engineering spent weeks on it. **Did the new page convert better — and is it worth shipping to 100% of traffic?**

- **H₀:** new page conversion == old page conversion
- **H₁:** new page **increases** conversion (one-sided ship decision; two-sided run as a robustness check)
- **Primary metric:** conversion rate · **Guardrail method:** demonstrated on Cookie Cats (D1 retention as canary; engagement depth via Welch's t-test — the exact pipeline AOV/load-time telemetry plugs into at a company that logs it)

## Datasets (real, no simulation)

| Dataset | Source | Size | What's inside |
|---|---|---|---|
| `data/ab_data.csv` | Udacity e-commerce A/B test (Kaggle mirror) | 294,478 rows | user_id, real timestamps (Jan 2017), old/new page, converted 0/1 |
| `data/countries.csv` | same experiment | paired rows | US/CA/UK — a real **pre-treatment** segmentation axis |
| `data/cookie_cats.csv` | Kaggle Cookie Cats | 90,189 users | gate_30 vs gate_40, D1/D7 retention, sum_gamerounds |

## Methodology

- **Power analysis first** (`NormalIndPower`, Cohen's h): 12.04% baseline, 1.0pp MDE, α = 0.05, 80% power → **13,557 users per variant**. The real test ran with ~145k per arm = **10.7× the requirement**.
- **Real data hygiene:** 3,893 group/page mismatch rows dropped (assignment contradicts served page — uninterpretable), 3,894 duplicate-user rows dropped (unit of analysis = user).
- **SRM check before any analysis** (χ² goodness-of-fit at ultra-strict α = 0.001): cleaned split 145,274/145,310 → pass.
- **Correct test per data type:** pooled two-proportion Z-test (binary conversion), Welch's t-test (continuous gamerounds — unequal variances never assumed), χ² twin (chi² = z² equivalence shown).
- **Effect sizes + CIs:** Cohen's h / d and 95% CIs on every difference; p-values alone shrink to 0 as n grows.

## Key Findings (actual outputs from the executed notebook)

- ✅ **SRM clean on both real experiments** (e-commerce χ² p = 0.947; Cookie Cats p = 0.009 > strict bar)
- 🚫 **Primary verdict — DO NOT SHIP:** old 12.04% vs new 11.88% (−0.16pp; relative −1.3%). One-sided p = 0.905, two-sided p = 0.190, 95% CI [−0.39%, +0.08%] — **excludes the +1.0pp ship threshold** at 10.7× required power. A properly powered *negative*, not an underpowered "maybe".
- 🎣 **Real peeking audit:** across 23 daily dashboard checks on real timestamps, cumulative p never dipped near α (range 0.46–0.94) — no early false win existed here; the one-readout discipline is still the takeaway.
- 🔁 **No novelty effect to decay:** week-1 lift −0.16pp ≈ week-2 lift −0.16pp (real dates).
- 🌎 **Real segment check (Simpson's):** UK +0.11pp vs US −0.22pp / CA −0.69pp — heterogeneous signs, no significant stratum → aggregate verdict stands.
- 🐱 **Cookie Cats (real, two-sided):** `retention_7` significantly better at gate_30 (19.02% vs 18.20%, **p = 0.0016**); `retention_1` quiet (p = 0.074); engagement guardrail flat (Welch t, p = 0.070) → **keep gate_30** — the D1/D7 conflict discussed as short-term friction vs durable value.

## What This Project Demonstrates (for recruiters / hiring managers)

| Skill | Where it is proven |
|---|---|
| **SQL / data engineering** | `sql/01_data_cleaning.sql` — staging → audit → dedup (`ROW_NUMBER`) → SRM chi-square → KPI aggregation, PostgreSQL dialect with MySQL/BigQuery notes |
| **Experimental design** | power analysis sized at n=13,557/arm *before* looking at outcomes; one Microsoft-style guardrail (SRM) checked at α=0.001 |
| **Statistics, correctly applied** | proportions z-test (two proportions, one-sided *ship* test + two-sided), Welch's t-test, confidence intervals, Cohen's h/d |
| **Pitfall forensics** | peeking audit over 23 real days, novelty-effect split (week 1 vs 2), Simpson's paradox country breakdown |
| **Productionisation** | reusable `ab_stats.py` library, deterministic `make_notebook.py`, shipped Streamlit app, BI-ready exports |

## Tech Stack

Python (Pandas · NumPy · SciPy · statsmodels · Matplotlib/Seaborn · Plotly) · SQL (PostgreSQL) · Streamlit · Power BI · Jupyter

## Repository Structure

```
ab-test-checkout-flow/
├── AB_Test_Checkout_Flow_Analysis.ipynb   ← core analysis (executed, real outputs)
├── ab_stats.py                             ← reusable, documented statistical library
├── app.py                                  ← Streamlit interactive demo (real datasets)
├── make_notebook.py                        ← rebuilds the notebook deterministically
├── sql/
│   └── 01_data_cleaning.sql                ← SQL extraction + cleaning + SRM + KPI queries
├── data/
│   ├── ab_data.csv                         ← real Udacity e-commerce test (294,478 rows)
│   ├── countries.csv                       ← real country mapping
│   └── cookie_cats.csv                     ← real Kaggle dataset (90,189 rows)
├── exports/                                ← Power BI-ready CSVs
│   ├── experiment_summary.csv              ← both experiments, per variant
│   ├── daily_conversion_trend.csv          ← REAL dates x variant conversion
│   ├── segment_metrics.csv                 ← real country/retention segment table
│   └── peeking_daily_pvalues.csv           ← REAL daily cumulative p-values
├── report.md                               ← 2-page executive report
├── POWERBI_BUILD_GUIDE.md                  ← step-by-step Power BI build (< 3 hours)
├── requirements.txt
└── .streamlit/config.toml                  ← dark theme
```

## How to Run

```bash
pip install -r requirements.txt

# 1. Core analysis (execute the notebook in Jupyter / VS Code / Kaggle)
jupyter lab AB_Test_Checkout_Flow_Analysis.ipynb

# 2. Interactive demo
streamlit run app.py

# 3. Dashboard: open Power BI Desktop, load exports/*.csv
#    Full click-by-click instructions: POWERBI_BUILD_GUIDE.md
```

## Deployment (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo (root contains `app.py`, `ab_stats.py`, `.streamlit/config.toml`).
2. https://share.streamlit.io → **New app** → repo/branch → main file `app.py` → Deploy.
3. Add the public URL here and to the links below.

## Links

- 📓 [Analysis Notebook (executed outputs)](AB_Test_Checkout_Flow_Analysis.ipynb)
- 🌐 Streamlit live demo: *(add URL after deploying)*
- 📊 Power BI dashboard: *(screenshot / PDF export — build guide included)*
- 📄 [Written Report for stakeholders](report.md)
