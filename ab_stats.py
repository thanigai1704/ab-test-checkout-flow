# =============================================================================
# ab_stats.py — Reusable A/B Testing Statistical Library
# =============================================================================
# Every function in this library is built to be INTERVIEW-DEFENSIBLE.
# Each docstring explains (a) what the test does, (b) why that specific test is
# the statistically correct choice, and (c) what the assumptions are.
#
# Design principle: an A/B test is a hypothesis test on random assignment.
# We never just "run a t-test because it exists" — every test here is chosen
# because the data type (binary proportion vs continuous metric) demands it.
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_effectsize,
    proportions_ztest,
    confint_proportions_2indep,
)


# -----------------------------------------------------------------------------
# 1. POWER ANALYSIS — "How many users do we need BEFORE we start?"
# -----------------------------------------------------------------------------
def run_power_analysis(
    baseline_rate: float,
    mde_pp: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "larger",
) -> dict:
    """
    Pre-experiment sample size calculation for a two-proportion A/B test.

    WHY THIS EXISTS (the fresher mistake this prevents):
    -------------------------------------------------
    Statistical power = P(detect a real effect | the effect is real).
    If you skip this step and just run "until something looks significant",
    you end up with an UNDER-POWERED test. An under-powered test doesn't just
    "take longer" — it actively produces false negatives. Example: with only
    ~1,000 users per arm, a true +1.5pp lift has only ~11% chance of being
    detected. You would ship "no effect" verdicts on tests that actually
    worked, and valid ideas die in the backlog. Power < 0.50 is especially
    dangerous: significance then becomes literally a coin flip on noise.

    HOW THE MATH WORKS:
    -------------------
    - proportion_effectsize(p1, p2) converts the two proportions into the
      standardised effect size Cohen's h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2)).
      (Arcsine transform stabilises the variance of a proportion, because
      variance of a Bernoulli depends on p itself.)
    - NormalIndPower.solve_power() inverts the power equation:
        power = Phi( |h| * sqrt(n/2) - z_alpha )  (for a one-sided test)
      and solves for n given the other three quantities.

    ASSUMPTION: independent samples, unequal-variance handled by the pooled /
    arcsine approximation, which is accurate for |h| in this regime. For very
    small rates (<1%) a simulation-based power check is recommended as a
    sanity check.

    Parameters
    ----------
    baseline_rate : control conversion rate, e.g. 0.12
    mde_pp        : minimum detectable effect in PERCENTAGE POINTS, e.g. 1.5
    alpha         : Type-I error cap (P(declare significant | no effect))
    power         : target power = 1 - beta (beta = Type-II error cap)
    alternative   : 'larger' (one-sided: new flow only needs to beat old)
                    or 'two-sided'.

    Returns
    -------
    dict with n per variant, Cohen's h, z_alpha, and the MDE in absolute terms.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1")
    treatment_rate = baseline_rate + mde_pp / 100.0
    if not 0 < treatment_rate < 1:
        raise ValueError("baseline + MDE crosses 0/1 boundary; pick a sane MDE")

    # Cohen's h: variance-stabilised effect size on the arcsine scale.
    h = proportion_effectsize(treatment_rate, baseline_rate)

    # Invert the normal-approx power formula to solve for n per arm.
    analysis = NormalIndPower()
    n_per_variant = analysis.solve_power(
        effect_size=h, alpha=alpha, power=power, alternative=alternative
    )

    return {
        "n_per_variant": int(np.ceil(n_per_variant)),
        "n_total": int(np.ceil(n_per_variant)) * 2,
        "cohen_h": h,
        "alpha": alpha,
        "power": power,
        "baseline_rate": baseline_rate,
        "treatment_rate_assumed": treatment_rate,
        "mde_pp": mde_pp,
        "alternative": alternative,
    }


def power_curve(
    baseline_rate: float,
    mde_pp_grid: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """Sample size vs. MDE curve. Used for the Section-2 plot: it shows the
    exponentially shrinking cost of detecting ever-smaller effects — the core
    trade-off conversation with a PM ('you want to detect 0.1pp? that needs
    2 million users per arm')."""
    rows = []
    for mde in mde_pp_grid:
        try:
            r = run_power_analysis(baseline_rate, float(mde), alpha, power)
            rows.append({"mde_pp": mde, "n_per_variant": r["n_per_variant"]})
        except ValueError:
            continue
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 2. SRM — "Is the random assignment actually random?"
# -----------------------------------------------------------------------------
def check_srm(
    counts: dict | pd.Series,
    expected_ratio: dict | None = None,
    alpha: float = 0.001,
) -> dict:
    """
    Sample Ratio Mismatch test: Chi-square goodness-of-fit on VARIANT COUNTS.

    WHY SRM BEFORE ANYTHING ELSE:
    ----------------------------
    Randomisation is the ONLY thing that lets us claim causality. If the split
    is not the ratio you designed (here 50/50), the experiment infrastructure
    is broken — and every downstream p-value, lift, CI is built on sand.
    Standard industry practice (Microsoft/Booking.com/Spotify playbooks) is to
    run SRM checks FIRST, and to use an ultra-strict alpha (0.001, not 0.05)
    because a mismatch is detectable with huge n — using 0.05 here would fire
    false alarms on perfectly fine experiments because the chi-square
    statistic grows linearly with n.

    NOTE the deliberate asymmetry: alpha=0.001 for a test we HOPE does NOT
    fire (a health check), alpha=0.05 for the main effect test. This is not
    hypocrisy — the two alphas protect against different failure modes.

    Parameters
    ----------
    counts          : observed counts per variant, e.g. {"control": 5012,
                      "treatment": 4988}
    expected_ratio  : expected proportions, default equal split.
    alpha           : decision threshold for the goodness-of-fit p-value.

    Returns
    -------
    dict(observed, expected, chi2, dof, p_value, sr_ratio, is_srm)
    """
    counts = pd.Series(counts, dtype=float)
    if expected_ratio is None:
        expected_ratio = {k: 1.0 / len(counts) for k in counts.index}
    expected_ratio = pd.Series(expected_ratio, dtype=float)

    # Scale the designed ratio to the same total volume as observed.
    expected_counts = counts.sum() * (expected_ratio / expected_ratio.sum())

    # Chi-square statistic: sum( (O-E)^2 / E ) over variants.
    chi2, p_value = stats.chisquare(
        f_obs=counts.values, f_exp=expected_counts.values
    )
    sr_ratio = counts.max() / counts.min() if counts.min() > 0 else np.inf

    return {
        "observed": counts.to_dict(),
        "expected": expected_counts.round(1).to_dict(),
        "chi2": chi2,
        "dof": len(counts) - 1,
        "p_value": p_value,
        "sample_ratio": sr_ratio,
        "is_srm": bool(p_value < alpha),  # True = RED FLAG: assignment broken
        "alpha": alpha,
    }


# -----------------------------------------------------------------------------
# 4. MAIN EFFECT TESTS
# -----------------------------------------------------------------------------
def run_ztest(
    successes_treatment: int,
    n_treatment: int,
    successes_control: int,
    n_control: int,
    alpha: float = 0.05,
    alternative: str = "larger",
) -> dict:
    """
    Two-proportion Z-test for the PRIMARY metric (conversion rate).

    WHY Z-TEST (not chi-square, not t-test):
    ---------------------------------------
    - Conversion data is binary (0/1). The sample proportion is a SUM of
      independent Bernoullis, which the CLT says converges to a normal
      distribution with known-variance structure (Var = p(1-p)). The Z-test
      uses exactly this: z = (p_t - p_c) / sqrt( p_pool(1-p_pool)(1/n_t + 1/n_c) ).
    - The POOLED variance (p_pool) under H0 is the right choice here: under
      H0 both variants share one rate, so we estimate it from the combined
      sample. This makes the test exact-ish and the classic 99% choice for
      online experiments.
    - Chi-square 2x2 IS mathematically ~ equivalent to the two-sided Z-test
      (chi2 == z^2), but Z gives signed direction + plays nicer with
      one-sided tests, so we use Z as the workhorse.
    - T-test on raw 0/1 values is also valid-ish (large n) but wasteful and
      muddier to defend. Welch's t-test is reserved for CONTINUOUS metrics
      (AOV, load time) below — different data, different test.

    alternative='larger': one-sided — we ship only if treatment BEATS control,
    never if it merely "changes" it. Industry scale-ups (e.g. Airbnb-style
    playbooks) favour one-sided for ship decisions; two-sided is the
    conservative academic default. We make the choice explicit, not silent.
    """
    count = np.array([successes_treatment, successes_control])
    nobs = np.array([n_treatment, n_control])

    z_stat, p_value = proportions_ztest(
        count, nobs, alternative=alternative
    )

    # Diff-in-proportions and its 95% CI (Newcombe-style via statsmodels).
    diff = successes_treatment / n_treatment - successes_control / n_control
    ci_low, ci_high = confint_proportions_2indep(
        count1=successes_treatment, nobs1=n_treatment,
        count2=successes_control, nobs2=n_control,
        method="wald", compare="diff", alpha=alpha,
    )

    # Cohen's h for proportions: 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2)).
    p_t = successes_treatment / n_treatment
    p_c = successes_control / n_control
    cohens_h = 2 * np.arcsin(np.sqrt(p_t)) - 2 * np.arcsin(np.sqrt(p_c))

    return {
        "rate_treatment": p_t,
        "rate_control": p_c,
        "abs_diff": diff,
        "rel_lift": diff / p_c,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohen_h": cohens_h,
        "is_significant": p_value < alpha,
        "n_treatment": n_treatment,
        "n_control": n_control,
        "alpha": alpha,
    }


def run_ttest(
    treatment_values: np.ndarray | pd.Series,
    control_values: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Welch's two-sample t-test for CONTINUOUS guardrail metrics (AOV, load).

    WHY WELCH (unequal variance), not Student's t:
    ---------------------------------------------
    1. Real-user behaviour is heterogeneous — heavy buyers vs one-off buyers
       means variance is NEVER equal across arms in practice. Welch does not
       assume it; Student's t does.
    2. Even when variances ARE equal, Welch loses almost nothing in power at
       realistic n. So it strictly dominates: same power, fewer assumptions.
    3. scipy.stats.ttest_ind(equal_var=False) = Welch: it adjusts both the
       statistic and (via Satterthwaite) the degrees of freedom.
    """
    t_stat, p_value = stats.ttest_ind(
        treatment_values, control_values, equal_var=False  # <-- Welch
    )
    sd_t = np.std(treatment_values, ddof=1)
    sd_c = np.std(control_values, ddof=1)
    mean_t = np.mean(treatment_values)
    mean_c = np.mean(control_values)

    # Cohen's d (pooled version for effect size reporting).
    pooled_sd = np.sqrt(((len(treatment_values) - 1) * sd_t**2
                         + (len(control_values) - 1) * sd_c**2)
                        / (len(treatment_values) + len(control_values) - 2))
    cohens_d = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else np.nan

    # Welch CI on the mean difference.
    se = np.sqrt(sd_t**2 / len(treatment_values) + sd_c**2 / len(control_values))
    dof = (sd_t**2/len(treatment_values) + sd_c**2/len(control_values))**2 / (
        (sd_t**2/len(treatment_values))**2/(len(treatment_values)-1)
        + (sd_c**2/len(control_values))**2/(len(control_values)-1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    return {
        "mean_treatment": mean_t,
        "mean_control": mean_c,
        "mean_diff": mean_t - mean_c,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": (mean_t - mean_c) - t_crit * se,
        "ci_high": (mean_t - mean_c) + t_crit * se,
        "cohen_d": cohens_d,
        "is_significant": p_value < alpha,
        "welch_dof": dof,
        "alpha": alpha,
    }


def guardrail_verdict(result: dict, protect_lower: bool = True) -> str:
    """
    Guardrail decisions: DEGRADE / IMPROVED / NO CHANGE.

    Guardrails are NOT a green light — they are a VETO test. A test passes on
    guardrails if the primary metric improved AND nothing got WORSE.
    protect_lower=True: only lower values than control are degradation
    (e.g. AOV); protect_lower=False: only HIGHER values hurt (e.g. load time).
    """
    passed = result["is_significant"]
    diff = result["mean_diff"]
    degraded = (diff < 0) if protect_lower else (diff > 0)
    if passed and degraded:
        return "DEGRADE (treatment significantly worse)"
    if passed:
        return "IMPROVED (significantly better)"
    if abs(diff) > 0:
        return "NO CHANGE (within noise)"
    return "NO CHANGE"
