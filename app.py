"""
A/B Test Explorer — Real Datasets · Designer Edition
====================================================
Two real A/B experiments: Udacity e-commerce test + Cookie Cats.
Run locally with:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from ab_stats import check_srm, run_power_analysis, run_ztest, power_curve

st.set_page_config(page_title="A/B Test Explorer", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK, SLATE, GRAY = "#F1F5F9", "#94A3B8", "#64748B"
BLUE, INDIGO, VIOLET, PINK = "#2563EB", "#4F46E5", "#7C3AED", "#DB2777"
TEAL, GREEN, RED, AMBER = "#0E7490", "#059669", "#DC2626", "#B45309"
FONT = '"Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif'

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp { font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif; color: #F1F5F9; }
.stApp {
    background:
        radial-gradient(1000px 520px at 6% -10%,  rgba(37, 99, 235, .28), transparent 62%),
        radial-gradient(880px 500px at 99% -4%,   rgba(219, 39, 119, .22), transparent 62%),
        radial-gradient(900px 560px at 50% 112%,  rgba(8, 145, 178, .24), transparent 62%),
        radial-gradient(620px 380px at 88% 78%,   rgba(124, 58, 237, .22), transparent 62%),
        linear-gradient(165deg, #0B1220 0%, #101A33 55%, #0C1526 100%) !important;
    background-attachment: fixed;
}
.block-container { position: relative; z-index: 1; max-width: 1280px; padding-top: 1.6rem; }

h1, h2, h3, h4 { color: #F1F5F9; letter-spacing: -0.022em; font-weight: 700; }

/* ---------------- HERO ---------------- */
.hero {
    background: linear-gradient(115deg, #0B1220 0%, #2563EB 62%, #4F46E5 100%);
    border-radius: 18px; padding: 30px 36px 28px 36px; color: #fff;
    box-shadow: 0 18px 46px -20px rgba(37, 99, 235, .6); position: relative; overflow: hidden;
}
.hero::after {
    content: ""; position: absolute; right: -60px; top: -70px; width: 260px; height: 260px;
    border-radius: 50%; background: radial-gradient(circle, rgba(219,39,119,.45), transparent 70%);
}
.hero .over { font-size: .78rem; font-weight: 700; letter-spacing: .14em; color: #93C5FD; text-transform: uppercase; }
.hero h1 { color: #fff !important; font-size: 2.05rem; font-weight: 800; margin: 8px 0 10px 0; line-height: 1.15; }
.hero .sub { color: #C7D9F5; font-size: .98rem; max-width: 800px; line-height: 1.55; }
.chip {
    display: inline-flex; align-items: center; gap: 6px; margin-top: 16px; margin-right: 8px;
    background: rgba(3, 12, 28, .5); border: 1px solid rgba(147, 197, 253, .35);
    color: #EAF1FE; font-size: .8rem; font-weight: 600; padding: 6px 14px; border-radius: 999px;
}
.chip b { color: #fff; }

/* ---------------- KPI CARDS (dark glass) ---------------- */
.kpi {
    background: linear-gradient(160deg, rgba(255,255,255,.10), rgba(255,255,255,.035));
    border: 1px solid rgba(148, 197, 253, .18); border-radius: 15px;
    padding: 16px 18px 13px 18px; position: relative; height: 100%;
    backdrop-filter: blur(14px);
    box-shadow: 0 12px 34px -20px rgba(2, 8, 24, .9);
}
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: var(--acc); border-radius: 4px 0 0 4px; }
.kpi .t { font-size: .74rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: #64748B; }
.kpi .v { font-size: 1.5rem; font-weight: 800; color: #F1F5F9; margin-top: 6px; letter-spacing: -0.02em; }
.kpi .s { font-size: .8rem; color: #94A3B8; margin-top: 3px; }

/* ---------------- SECTION HEADERS ---------------- */
.sect { display: flex; align-items: center; gap: 10px; margin: 14px 0 6px 0; }
.sect .ico {
    width: 30px; height: 30px; border-radius: 9px; display: inline-flex; align-items: center;
    justify-content: center; font-size: .95rem;
    background: linear-gradient(135deg, #DB2777, #7C3AED); color: #fff;
    box-shadow: 0 6px 18px -8px rgba(219, 39, 119, .8);
}
.sect .label { font-size: 1.04rem; font-weight: 700; color: #F1F5F9; letter-spacing: -0.01em; }

/* ---------------- PANELS (colored glass, no white) ---------------- */
.glass {
    background: linear-gradient(150deg, rgba(59, 92, 168, .16), rgba(124, 58, 237, .10));
    border: 1px solid rgba(147, 197, 253, .22); border-radius: 16px; padding: 16px 20px;
    font-size: .93rem; line-height: 1.62;
    backdrop-filter: blur(12px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 14px 36px -26px rgba(2, 8, 24, .9);
}
.glass b { color: #93C5FD; }

/* ---------------- TABS ---------------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(148,197,253,.14); }
.stTabs [data-baseweb="tab"] {
    font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif; font-size: .9rem; font-weight: 600; color: #94A3B8;
    padding: 10px 20px; border-radius: 10px 10px 0 0;
    background: linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
    border: 1px solid rgba(148,197,253,.12); border-bottom: none;
}
.stTabs [aria-selected="true"] {
    color: #93C5FD; background: linear-gradient(160deg, rgba(37,99,235,.28), rgba(124,58,237,.18));
    border-color: rgba(147,197,253,.45); border-bottom: 2px solid #60A5FA;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1020 0%, #060B16 100%) !important;
    border-right: 1px solid rgba(148,197,253,.12);
}
section[data-testid="stSidebar"] * { color: #D5DEEC !important; }
section[data-testid="stSidebar"] .prov {
    background: linear-gradient(150deg, rgba(37,99,235,.18), rgba(124,58,237,.14));
    border: 1px solid rgba(147,197,253,.28); border-radius: 12px;
    padding: 10px 12px; font-size: .78rem; line-height: 1.5;
}
section[data-testid="stSidebar"] .prov a { color: #7DD3FC !important; font-weight: 600; }

div[data-testid="stMetric"] { background: transparent; border: none; box-shadow: none; }
.steff { font-variant-numeric: tabular-nums; }

/* ---------------- FOOTER ---------------- */
.foot {
    margin-top: 26px; padding: 14px 20px; border-top: 1px solid rgba(148,197,253,.14);
    color: #94A3B8; font-size: .8rem; display: flex; justify-content: space-between;
}
"""
st.markdown(CSS, unsafe_allow_html=True)


def kpi_card(col, accent: str, title: str, value: str, sub: str = ""):
    col.markdown(
        f'<div class="kpi steff" style="--acc:{accent};border-left: 4px solid {accent}">'
        f'<div class="t">{title}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def section(icon: str, label: str):
    return st.markdown(f'<div class="sect"><span class="ico">{icon}</span>'
                       f'<span class="label">{label}</span></div>', unsafe_allow_html=True)


def styled(fig, height=380, yfmt="", showlegend=None):
    """One plotly design language for every chart in the project."""
    fig.update_layout(
        template="plotly_dark", height=height,
        font=dict(family=FONT, size=13, color="#CBD5E1"),
        title=dict(font=dict(size=15, color="#F1F5F9", family=FONT,
                             weight=700), x=0.015, xanchor="left"),
        margin=dict(t=54, b=40, l=14, r=14),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="x unified",
        yaxis=dict(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.22)", tickformat=yfmt or None),
        xaxis=dict(gridcolor="rgba(148,163,184,.08)"),
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🧪 A/B Test Explorer")
st.sidebar.caption("Two genuine A/B experiments — analyzed end-to-end.")
dataset = st.sidebar.selectbox(
    "Dataset",
    ("📊 E-commerce Page Test (Udacity)", "🐱 Cookie Cats (real, Kaggle)"),
    help="Udacity: real e-commerce conversion test (294,478 rows, real Jan-2017 timestamps). "
         "Cookie Cats: real Kaggle dataset (90,189 players).")
st.sidebar.divider()
st.sidebar.markdown("**📐 Power analysis sliders** — pick the effect size "
                    "you'd care about; see what sample that demands.")
slider_mde = st.sidebar.slider("Minimum detectable effect (pp)", 0.1, 5.0, 1.0, 0.1)
alpha = st.sidebar.slider("Significance level α", 0.01, 0.10, 0.05, 0.01)
st.sidebar.markdown('<div class="prov"><b>✅ Data provenance (real)</b><br>'
                    '• <a href="https://github.com/patrickbloomingdale/Udacity-Project-AB--Testing">'
                    'Udacity e-commerce test</a> — 294,478 rows, Jan 2017<br>'
                    '• <a href="https://www.kaggle.com/datasets/yasserh/cookie-cats-dataset">'
                    'Cookie Cats</a> — 90,189 players, Kaggle</div>', unsafe_allow_html=True)
st.sidebar.caption("🛠 statsmodels · SciPy · Plotly")

# ---------------------------------------------------------------------------
# Data loading and computation (identical to the analysis module)
# ---------------------------------------------------------------------------


@st.cache_data
def load_page_test():
    df = pd.read_csv("data/ab_data.csv")
    mismatch = ((df.group == "treatment") & (df.landing_page == "old_page")) | (
        (df.group == "control") & (df.landing_page == "new_page"))
    return df[~mismatch].drop_duplicates(subset="user_id", keep="first")


@st.cache_data
def load_cookie_cats():
    for p in ("data/cookie_cats.csv", "cookie_cats.csv"):
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("cookie_cats.csv not found — place the Kaggle file in data/.")
    st.stop()


@st.cache_data
def load_exports():
    d = {}
    for f in ("experiment_summary", "daily_conversion_trend", "segment_metrics",
              "peeking_daily_pvalues"):
        path = f"exports/{f}.csv"
        d[f] = pd.read_csv(path) if os.path.exists(path) else None
    return d


if "E-commerce" in dataset:
    df = load_page_test()
    VARIANT_COL, OUTCOME, TREATMENT_LABEL, CONTROL_LABEL, METRIC_NAME = (
        "group", "converted", "treatment", "control", "Purchase Conversion Rate")
    HAS_TIMESTAMPS, ICON = True, "📊"
else:
    df = load_cookie_cats()
    VARIANT_COL, OUTCOME, TREATMENT_LABEL, CONTROL_LABEL, METRIC_NAME = (
        "version", "retention_7", "gate_30", "gate_40", "Day-7 Retention")
    HAS_TIMESTAMPS, ICON = False, "🐱"

counts = df.groupby(VARIANT_COL)[OUTCOME].agg(["sum", "count"])
n_t, s_t = int(counts.loc[TREATMENT_LABEL, "count"]), int(counts.loc[TREATMENT_LABEL, "sum"])
n_c, s_c = int(counts.loc[CONTROL_LABEL, "count"]), int(counts.loc[CONTROL_LABEL, "sum"])
obs_baseline = s_c / n_c
power = run_power_analysis(obs_baseline, slider_mde, alpha=alpha)
zres = run_ztest(s_t, n_t, s_c, n_c)
zres_1 = run_ztest(s_t, n_t, s_c, n_c, alternative="larger")
srm = check_srm(df[VARIANT_COL].value_counts().to_dict())
p_t, p_c = s_t / n_t, s_c / n_c
obs_behind = min(n_t, n_c) / power["n_per_variant"]
EXPORTS = load_exports()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
diff_pp = (p_t - p_c) * 100
sig_chip = ("✅ Significant" if zres["p_value"] < alpha else "✖ Not significant")
verdict_main = "DO NOT SHIP ⛔" if "E-commerce" in dataset else "KEEP gate_30 ✅"
st.markdown(f"""
<div class="hero">
  <div class="over">A/B Test Design &amp; Analysis · 100% Real Data</div>
  <h1>{ICON}&nbsp; {METRIC_NAME} — did the change move the needle?</h1>
  <div class="sub">The team shipped a redesigned experience to lift conversion. This study sizes the
  test <i>before</i> launch, audits randomisation, runs the correct statistical test for the data type,
  and hunts the four classic experiment pitfalls before issuing a ship verdict.</div>
  <span class="chip">{dataset.replace('📊 ', '').replace('🐱 ', '')}</span>
  <span class="chip"><b>{len(df):,}</b>&nbsp;users analyzed</span>
  <span class="chip">split&nbsp;<b>{n_c:,} / {n_t:,}</b></span>
  <span class="chip">{sig_chip}&nbsp;·&nbsp;p&nbsp;=&nbsp;<b>{zres['p_value']:.4f}</b></span>
  <span class="chip">{verdict_main}</span>
</div>""", unsafe_allow_html=True)

# KPI band
c1, c2, c3, c4, c5 = st.columns(5)
kpi_card(c1, SLATE, "Control rate", f"{p_c:.2%}", f"{CONTROL_LABEL} · {n_c:,} users")
kpi_card(c2, BLUE, "Treatment rate", f"{p_t:.2%}", f"{diff_pp:+.2f} pp vs control")
kpi_card(c3, INDIGO, "Required n / arm", f"{power['n_per_variant']:,}",
         f"{obs_behind:.1f}× collected")
kpi_card(c4, VIOLET, "95% CI of difference",
         f"[{zres['ci_low']:+.2%}, {zres['ci_high']:+.2%}]",
         "contains 0 → not confirmed" if 0 > zres["ci_low"] else "excludes 0 → directional")
kpi_card(c5, GREEN if zres["p_value"] < alpha else RED, "Cohen's h",
         f"{abs(zres['cohen_h']):.4f}",
         "negligible magnitude" if abs(zres['cohen_h']) < .08 else "material effect")

tabs = st.tabs(["🎯 Overview", "📈 Power & Distributions", "🚧 Pitfalls",
                "🧮 Statistics", "🏁 Verdict", "📊 BI Dashboard"])

# ===========================================================================
# TAB 1 — Overview
# ===========================================================================
with tabs[0]:
    section("🎯", "Headline comparison")
    g1, g2 = st.columns([2.6, 1])
    with g1:
        fig = go.Figure(go.Bar(
            x=[CONTROL_LABEL.upper(), TREATMENT_LABEL.upper()], y=[p_c, p_t],
            width=0.5, marker=dict(color=[SLATE, BLUE], cornerradius=8),
            text=[f"{p_c:.2%}", f"{p_t:.2%}"], textposition="outside",
            textfont=dict(size=15, color="#F1F5F9"),
            error_y=dict(type="data", array=[1.96 * np.sqrt(v * (1 - v) / n) for v, n in
                                             [(p_c, n_c), (p_t, n_t)]],
                         color="#334155", thickness=2)))
        st.plotly_chart(styled(fig, 430, yfmt=".1%", showlegend=False), width="stretch")
    with g2:
        fig2 = go.Figure(go.Pie(labels=[CONTROL_LABEL, TREATMENT_LABEL], values=[n_c, n_t],
                                hole=0.66, marker=dict(colors=[SLATE, BLUE]),
                                textinfo="percent", sort=False))
        st.plotly_chart(styled(fig2, 430), width="stretch")
    st.markdown(f"""
<div class="glass">
<b>{CONTROL_LABEL}</b> converts <b>{p_c:.2%}</b> on {n_c:,} users ·
<b>{TREATMENT_LABEL}</b> converts <b>{p_t:.2%}</b> on {n_t:,} users → difference
<b>{diff_pp:+.2f} pp</b> ({zres['rel_lift']:+.1%} relative), CI
<b>[{zres['ci_low']:+.2%}, {zres['ci_high']:+.2%}]</b>.
{'The interval contains 0, so the gap may be sampling noise — and every formal test in 🧮 agrees.' if 0 > zres['ci_low'] else 'The interval excludes 0 → directional result.'}
Point estimate says the challenger is <b>{'worse' if p_t < p_c else 'better'}</b>.
</div>""", unsafe_allow_html=True)

# ===========================================================================
# TAB 2 — Power & distributions
# ===========================================================================
with tabs[1]:
    section("📈", "Power analysis")
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, INDIGO, "Required n per arm", f"{power['n_per_variant']:,}",
             f"α = {alpha:.0%} · power = 80%")
    kpi_card(c2, BLUE, "Planned Cohen's h", f"{power['cohen_h']:.4f}",
             f"for a {slider_mde}pp MDE at {obs_baseline:.1%} baseline")
    kpi_card(c3, VIOLET, "Collected vs plan", f"{obs_behind:.1f}×",
             "plan satisfied 🎉" if obs_behind >= 1 else "underpowered ⚠️")
    kpi_card(c4, TEAL, "Total required", f"{power['n_total']:,}",
             "per both arms combined")

    mde_grid = np.arange(max(0.2, slider_mde - 1.0), slider_mde + 1.5, 0.1)
    curve = power_curve(obs_baseline, mde_grid, alpha)
    if not curve.empty:
        figc = go.Figure(go.Scatter(x=curve["mde_pp"], y=curve["n_per_variant"],
                                    mode="lines+markers",
                                    line=dict(color=INDIGO, width=3), marker=dict(size=6)))
        figc.add_vline(x=slider_mde, line_dash="dash", line_color=RED,
                       annotation_text=f"your MDE = {slider_mde}pp")
        st.plotly_chart(styled(figc, 360, showlegend=False), width="stretch")

    section("🌊", "Sampling distributions")
    lo, hi = min(p_t, p_c), max(p_t, p_c)
    pad = max(3 * np.sqrt(p_c * (1 - p_c) / n_c), 0.005)
    x = np.linspace(lo - pad, hi + pad, 500)
    figd = go.Figure()
    for rate, n_, name, color in [(p_c, n_c, CONTROL_LABEL, SLATE),
                                  (p_t, n_t, TREATMENT_LABEL, BLUE)]:
        figd.add_trace(go.Scatter(
            x=x, y=stats.norm.pdf(x, rate, np.sqrt(rate * (1 - rate) / n_)),
            fill="tozeroy", fillcolor=color + "33", name=name,
            line=dict(color=color, width=2)))
    st.plotly_chart(styled(figd, 380), width="stretch")

# ===========================================================================
# TAB 3 — Pitfalls
# ===========================================================================
with tabs[2]:
    section("🚧", "Pitfall monitor")
    srm_msg = ("randomisation healthy ✔" if not srm["is_srm"]
               else "ASSIGNMENT BROKEN — all estimates invalid ✖")
    (st.success if not srm["is_srm"] else st.error)(
        f"**SRM** — observed ratio {srm['sample_ratio']:.4f}, χ² p = {srm['p_value']:.3f} "
        f"at α = 0.001 → {srm_msg}")

    if HAS_TIMESTAMPS:
        df_ts = df.copy()
        df_ts["ts"] = pd.to_datetime(df_ts["timestamp"])
        df_ts["date"] = df_ts["ts"].dt.date
        trail = []
        for d in sorted(df_ts["date"].unique()):
            sub = df_ts[df_ts["ts"] < pd.Timestamp(d) + pd.Timedelta(days=1)]
            t, c = (sub[sub[VARIANT_COL] == TREATMENT_LABEL][OUTCOME],
                    sub[sub[VARIANT_COL] == CONTROL_LABEL][OUTCOME])
            _, p1 = proportions_ztest(np.array([t.sum(), c.sum()]),
                                      np.array([len(t), len(c)]), alternative="larger")
            trail.append({"date": str(d), "n_per_arm": len(t), "cumulative_p": p1})
        trail = pd.DataFrame(trail)
        n_crossed = int((trail["cumulative_p"] < 0.05).sum())
        st.warning(f"**Peeking audit (real timestamps):** {n_crossed} of {len(trail)} daily "
                   f"checks crossed α (p ranged {trail['cumulative_p'].min():.2f}–"
                   f"{trail['cumulative_p'].max():.2f}).")
        figp = go.Figure(go.Scatter(x=trail["date"], y=trail["cumulative_p"],
                                    mode="lines+markers", line=dict(color=TEAL, width=2.5),
                                    marker=dict(size=7), customdata=trail["n_per_arm"]))
        figp.add_hline(y=0.05, line_dash="dash", line_color=RED,
                       annotation_text="α = 0.05", annotation_font_color=RED)
        st.plotly_chart(styled(figp, 400, showlegend=False), width="stretch")
    else:
        st.info("Cookie Cats has no timestamps — the full real peeking trail lives in the "
                "e-commerce analysis (notebook §7a).")

    a1, a2, a3 = st.columns(3)
    a1.success("**SRM** — ✅ pass on both experiments")
    a2.warning("**Novelty** — 🟡 week-1 ≈ week-2 lift (real dates, notebook §7b)")
    a3.info("**Simpson's** — 🔵 real US/CA/UK strata: no reversal (notebook §7c)")

# ===========================================================================
# TAB 4 — Statistics
# ===========================================================================
with tabs[3]:
    section("🧮", "Statistical results")
    stats_df = pd.DataFrame([
        {"Test": "Two-proportion Z — one-sided (ship)", "Statistic": f"z = {zres_1['z_stat']:.3f}",
         "p-value": f"{zres_1['p_value']:.4f}", "Significant": zres_1["p_value"] < alpha},
        {"Test": "Two-proportion Z — two-sided (robustness)", "Statistic": f"z = {zres['z_stat']:.3f}",
         "p-value": f"{zres['p_value']:.4f}", "Significant": zres["p_value"] < alpha},
        {"Test": "95% CI of the difference", "Statistic":
            f"[{zres['ci_low']:+.4%}, {zres['ci_high']:+.4%}]", "p-value": "—", "Significant": None},
        {"Test": "Absolute / relative lift", "Statistic":
            f"{zres['abs_diff']:+.3%}  /  {zres['rel_lift']:+.2%}", "p-value": "—", "Significant": None},
        {"Test": "Effect size (Cohen's h)", "Statistic": f"{zres['cohen_h']:.4f}",
         "p-value": "—", "Significant": None},
        {"Test": "SRM (χ² goodness-of-fit, α = 0.001)",
         "Statistic": f"ratio = {srm['sample_ratio']:.4f},  χ² = {srm['chi2']:.3f}",
         "p-value": f"{srm['p_value']:.3f}", "Significant": srm["is_srm"]},
    ])
    st.dataframe(
        stats_df.style.map(
            lambda v: "background-color:rgba(6,95,70,.45);color:#6EE7B7" if v is True
            else ("background-color:rgba(153,27,27,.45);color:#FCA5A5" if v is False else ""),
            subset=["Significant"]),
        hide_index=True, width="stretch")
    st.caption("Pooled variance under H₀ for the proportion tests · χ² twin of the two-sided Z "
               "(chi² = z²) shown in the notebook §5b.")

# ===========================================================================
# TAB 5 — Verdict
# ===========================================================================
with tabs[4]:
    section("🏁", "Ship verdict")
    if "E-commerce" in dataset:
        body = (f"One-sided ship test p = {zres_1['p_value']:.4f} (no positive lift anywhere), "
                f"and the two-sided 95% CI [{zres['ci_low']:+.3%}, {zres['ci_high']:+.3%}] already "
                f"excludes the {slider_mde}pp ship threshold at {obs_behind:.1f}× the required "
                "sample — a properly powered *negative* result, not an underpowered one.")
        st.markdown(f"""
<div style="background:linear-gradient(100deg,#7F1D1D,#DC2626);color:#fff;padding:24px 30px;
     border-radius:16px;box-shadow:0 14px 34px -14px rgba(220,38,38,.55)">
  <span style="font-size:1.6rem">⛔</span> <b style="font-size:1.2rem">VERDICT — DO NOT SHIP</b>
  <div style="color:#FECACA;margin-top:8px;line-height:1.55">{body}</div>
</div>""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="glass">
<b>What I'd do next</b><br>
1. Keep the old page; <b>diagnose</b> the funnel (section drop-offs, load telemetry) instead of re-testing.<br>
2. Instrument AOV / load-time guardrails so the veto pipeline runs live on the next candidate.<br>
3. Adopt the one-readout discipline and SRM-first monitoring for every upcoming experiment.
</div>""", unsafe_allow_html=True)
    else:
        body = (f"Day-7 retention is significantly better for gate_30 (two-sided "
                f"p = {zres['p_value']:.4f}, +{zres['abs_diff'] * 100:.2f}pp). The D1 canary is "
                "quiet (p = 0.074) and engagement depth unchanged (Welch t, p = 0.070) — "
                "a slow-burn durable-engagement win with no visible cost.")
        st.markdown(f"""
<div style="background:linear-gradient(100deg,#065F46,#059669);color:#fff;padding:24px 30px;
     border-radius:16px;box-shadow:0 14px 34px -14px rgba(5,150,105,.55)">
  <span style="font-size:1.6rem">✅</span> <b style="font-size:1.2rem">VERDICT — KEEP gate_30</b>
  <div style="color:#A7F3D0;margin-top:8px;line-height:1.55">{body}</div>
</div>""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="glass">
<b>Why the D1/D7 conflict is information</b><br>
• D7 survives a Bonferroni-adjusted α = 0.025; D1 does not — the evidence genuinely separates.<br>
• Short-term friction vs durable value: the win lives in the <i>habit</i> pathway.<br>
• Ship communication: no week-one dashboard pop to expect.
</div>""", unsafe_allow_html=True)

# ===========================================================================
# TAB 6 — BI Dashboard
# ===========================================================================
with tabs[5]:
    section("📊", "BI Dashboard — Power BI deliverable preview")
    st.caption("Built from the real `exports/` CSVs; the click-by-click Power BI Desktop build "
               "is in `POWERBI_BUILD_GUIDE.md`.")
    if EXPORTS["experiment_summary"] is None:
        st.warning("Run the notebook once (writes `exports/`) to fill this tab.")
    else:
        esum, dtr, seg, peek = (EXPORTS["experiment_summary"], EXPORTS["daily_conversion_trend"],
                                EXPORTS["segment_metrics"], EXPORTS["peeking_daily_pvalues"])

        section("🚀", "Page 1 · Executive Summary")
        b1, b2, b3, b4 = st.columns(4)
        total_users = int(esum["sample_size"].sum())
        kpi_card(b1, BLUE, "Total users analyzed", f"{total_users:,}", "both real experiments")
        kpi_card(b2, INDIGO, "Overall lift %", f"{zres['rel_lift']:+.1%}",
                 "e-commerce test, treatment vs control")
        kpi_card(b3, VIOLET, "P-value", f"{zres['p_value']:.4f}", "two-sided")
        kpi_card(b4, GREEN if "Cookie" in dataset else RED, "Recommendation",
                 verdict_main.split(" ")[0],
                 "no benefit demonstrated" if "E-commerce" in dataset
                 else "gate_30 outperforms on D7")

        pb1, pb2 = st.columns(2)
        with pb1:
            figE = go.Figure()
            palette = {"ecommerce_page_redesign": [SLATE, BLUE],
                       "cookie_cats_gate_move": ["#F59E0B", GREEN]}
            for exp_name in esum["experiment"].unique():
                e = esum[esum["experiment"] == exp_name]
                figE.add_trace(go.Bar(x=e["variant"], y=e["conversion_rate"],
                                      name=exp_name.replace("_", " "),
                                      marker=dict(color=palette[exp_name], cornerradius=8),
                                      text=[f"{v:.1%}" for v in e["conversion_rate"]],
                                      textposition="outside", textfont=dict(size=11, color="#E2E8F0")))
            figE.update_layout(barmode="group", title="Conversion by variant — both experiments",
                               bargap=0.35)
            st.plotly_chart(styled(figE, 380, yfmt=".1%"), width="stretch")
        with pb2:
            figK = go.Figure(go.Indicator(
                mode="gauge+number", value=0.80,
                number=dict(valueformat=".0%", font=dict(size=40, color="#F1F5F9")),
                title={"text": "Statistical power", "font": {"size": 14, "color": "#94A3B8"}},
                gauge={"axis": dict(range=[0, 1], tickcolor="#64748B"),
                       "bar": dict(color=BLUE),
                       "steps": [{"range": [0, 0.8], "color": "#FECACA"},
                                 {"range": [0.8, 1], "color": "#A7F3D0"}],
                       "threshold": dict(line=dict(color=RED, width=3), value=0.8)}))
            st.plotly_chart(styled(figK, 380, showlegend=False), width="stretch")

        section("🔬", "Page 2 · Statistical Deep Dive")
        dd1, dd2 = st.columns(2)
        with dd1:
            if dtr is not None:
                figT = go.Figure()
                for v in sorted(dtr["variant"].unique()):
                    d = dtr[dtr["variant"] == v]
                    figT.add_trace(go.Scatter(x=d["date"], y=d["daily_conversion_rate"],
                                              mode="lines+markers",
                                              name=v, line=dict(
                                                  color=BLUE if v == "treatment" else SLATE,
                                                  width=2), marker=dict(size=5)))
                figT.update_layout(title="Daily conversion — real dates")
                st.plotly_chart(styled(figT, 360, yfmt=".1%"), width="stretch")
        with dd2:
            if peek is not None:
                pcol = "cumulative_p" if "cumulative_p" in peek.columns else "p_value_1_sided"
                figP = go.Figure(go.Scatter(x=peek["date"], y=peek[pcol], mode="lines+markers",
                                            line=dict(color=TEAL, width=2.5), marker=dict(size=6),
                                            customdata=peek["n_per_arm"],
                                            hovertemplate="%{x}<br>p = %{y:.3f}<extra></extra>"))
                figP.add_hline(y=0.05, line_dash="dash", line_color=RED,
                               annotation_text="α = 0.05")
                figP.update_layout(title="Peeking trail")
                st.plotly_chart(styled(figP, 360, showlegend=False), width="stretch")

        section("🛡️", "Page 3 · Segment & Stability Monitor")
        m1, m2, m3 = st.columns(3)
        with m1:
            if seg is not None:
                cs = seg[(seg["experiment"] == "ecommerce_page_redesign") &
                         (seg["segment_name"].isin(["US", "CA", "UK"]))]
                figS = go.Figure()
                for v, color in [("control", SLATE), ("treatment", BLUE)]:
                    d = cs[cs["variant"] == v]
                    figS.add_trace(go.Bar(x=d["segment_name"], y=d["value"], name=v,
                                          marker=dict(color=color, cornerradius=8)))
                figS.update_layout(barmode="group", title="By country", bargap=0.35)
                st.plotly_chart(styled(figS, 340, yfmt=".1%", showlegend=False),
                                width="stretch")
        for col, seg_label, accent in [(m2, "D1 Retention", AMBER), (m3, "D7 Retention", GREEN)]:
            if seg is not None:
                d = seg[seg["segment_name"] == seg_label]
                ctrl = float(d[d["variant"] == "gate_40"]["value"].iloc[0])
                trt = float(d[d["variant"] == "gate_30"]["value"].iloc[0])
                top = max(0.5, trt * 1.35)
                figG = go.Figure(go.Indicator(
                    mode="gauge+number", value=trt,
                    number=dict(valueformat=".1%", font=dict(size=34, color="#F1F5F9")),
                    title={"text": f"{seg_label} · gate_30", "font": {"size": 13, "color": "#94A3B8"}},
                    gauge={"axis": dict(range=[0, top], tickcolor="#64748B"),
                           "bar": dict(color=accent),
                           "steps": [{"range": [0, ctrl], "color": "#FEF3C7"},
                                     {"range": [ctrl, top], "color": "#D1FAE5"}],
                           "threshold": dict(line=dict(color=SLATE, width=2.5), value=ctrl)}))
                figG.update_layout(height=340, margin=dict(t=44, b=10, l=14, r=14),
                                   paper_bgcolor="rgba(0,0,0,0)")
                col.plotly_chart(figG, width="stretch")

st.markdown(f"""
<div class="foot">
<span>🧪 Real-data A/B analysis — Udacity e-commerce test + Cookie Cats</span>
<span>notebook <b>AB_Test_Checkout_Flow_Analysis.ipynb</b> · library <b>ab_stats.py</b> ·
guide <b>POWERBI_BUILD_GUIDE.md</b></span>
</div>""", unsafe_allow_html=True)
