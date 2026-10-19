# A/B Test Final Report — E-commerce Page Redesign

**Experiment:** old purchase-funnel page vs redesigned page · 290,584 real users over 23 days (Jan 2017)
**Companion analysis:** Cookie Cats gate test (real, 90,189 players) for guardrail methodology
**Methodology:** pre-registered power analysis · SRM check · two-proportion Z-tests · real-timestamp peeking audit · segment checks

*(Deliverable 4 for the portfolio project: paste into Word/Google Docs and export as PDF. All numbers below are actual executed outputs on the real datasets.)*

---

## 1. Executive Summary

We tested the redesigned purchase page against the current one on **290,584 real customers** — more than 10× the sample we needed — and the new page **did not convert better; it converted slightly worse** (−0.16 percentage points). Because this test had overwhelming statistical power, we can say with confidence that *any improvement worth shipping (1 point or more) would already have shown up — it didn't.* Recommendation: **do not ship the redesign; invest instead in diagnosing why the conversion journey stalls** (page diagnostics, drop-off telemetry).

## 2. Hypothesis & Methodology

- **Hypothesis:** the redesigned page increases purchase conversion; only an increase counts (directional ship test).
- **Pre-agreed standard:** reliably detect a **+1.0pp** improvement (12.0% → 13.0%) with α = 0.05, power = 80% → required sample 13,557 users per arm.
- **Data hygiene (real defects, fixed before analysis):** 3,893 rows where the assignment label contradicted the page actually served were removed (uninterpretable for a causal verdict), plus 3,894 duplicate-user rows — analysis runs on one row per user (290,584).
- **Randomisation integrity (SRM):** the cleaned 50/50 split verified at ultra-strict tolerance (χ² p = 0.947) — the experiment infrastructure is trusted.
- **Tests used:** two-proportion Z-test (pooled variance, one-sided for the ship decision + two-sided as robustness check); the 2×2 chi-square is reported as a mathematical cross-check (p = 0.192 ≈ z-test's 0.190, as theory dictates).

## 3. Key Results

| Metric | Old page | New page | Difference | Verdict |
|---|---|---|---|---|
| **Conversion rate (primary)** | 12.04% | 11.88% | **−0.16pp** (relative −1.3%) | 1-sided p = 0.905; 2-sided p = 0.190 — **not significant, direction negative** |
| 95% CI of the difference | — | — | [−0.39%, +0.08%] | **Excludes the +1.0pp ship threshold entirely** |
| Effect size (Cohen's h) | — | — | 0.005 | Negligible |

*(Supporting chart 1: conversion rate by variant with 95% CI error bars — Notebook §4b. Supporting chart 2: real daily conversion trend, both arms hugging at ~12% — Notebook §4c.)*

Translation for the business: at ~12,600 users/day, a −0.16pp shift is on the order of **7,300 fewer orders a year (≈ −$320K at a $44 AOV)** — against no evidence of gain and the real cost of maintaining a second page build.

## 4. Pitfalls Checked & Findings

- **"Peeking" audit (real timestamps):** we replayed every daily dashboard check a PM would have made across all 23 real experiment days. The cumulative p-value **never came close to the 0.05 line** (range 0.46–0.94) — no false "win" was available in this data to tempt an early stop. The discipline stands: a single pre-committed readout, because the same instinct on luckier data manufactures fake wins.
- **Novelty / decay check (real dates):** week-1 lift −0.16pp vs week-2 lift −0.16pp — no early enthusiasm that faded; the page was never better.
- **Segment consistency (Simpson's paradox check, real US/CA/UK data):** UK is nominally positive (+0.11pp) while US (−0.22pp) and CA (−0.69pp) are negative — signs differ across segments but none is significant in isolation. No actionable segment win; the aggregate verdict stands.
- **Companion guardrail case (Cookie Cats, real):** D7 retention significantly favours the earlier gate (19.02% vs 18.20%, p = 0.0016), while the D1 canary is quiet (p = 0.074) and engagement depth (Welch's t-test on log-gamerounds, p = 0.070) shows no meaningful shift — the exact structure a real ship review uses to check "did we win X without breaking Y?"

## 5. Final Recommendation

**DO NOT SHIP.** The test was properly powered (helped by 10.7× the required sample) and the result is a genuine *negative* finding: the redesigned page yields no conversion benefit, its point estimate is harmful, and the confidence interval already rules out any ship-worthy lift. Extending the test would change nothing meaningful — it would only lower the bar for noise.

## 6. Next Steps

1. **Keep the old page live** (and the Cookie Cats-style secondary verdict: keep the gate at level 30).
2. **Diagnose, don't re-test:** invest in funnel diagnostics — section-level drop-off, load-time telemetry, scroll/heatmap data — to find where purchase intent is lost.
3. **Instrument the funnel for next time:** log AOV and load-time per variant so future ship reviews run the full guardrail-veto pipeline demonstrated on the Cookie Cats data.
4. **Ship process change, not the page:** adopt the one-readout discipline and SRM-first monitoring for every upcoming experiment.
