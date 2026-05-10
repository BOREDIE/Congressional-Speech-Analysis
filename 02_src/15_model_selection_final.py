"""
Final model selection:

Part A — All-subset confounder model comparison (2^5 = 32 models)
  Speaker-level DiD with every combination of the 5 DAG confounders.
  Select best model by AIC; verify DiD coefficient stability.

Part B — LMM vs OLS comparison (word-level)
  Linear Mixed Model with speaker random effect vs M5 (speaker FE).
  Tests whether the main finding holds under a more efficient estimator.
"""

from pathlib import Path
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

REPO    = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "03_output" / "did_results"
FIG_DIR = REPO / "03_output" / "figures"


# ---------------------------------------------------------------------------
# Build speaker-level dataset
# ---------------------------------------------------------------------------

def build_speaker_df(df, matched):
    """Aggregate word-level data to speaker × period, attach confounders from matched_sample."""
    def wmean(g):
        """Occurrence-count-weighted mean Y for a group."""
        w = g["occurrence_count"].clip(lower=1)
        return np.average(g["Y"], weights=w)

    agg = (df.groupby(["bioguide_id", "group", "aligned_period"], dropna=False)
             .apply(wmean).reset_index(name="Y_bar"))
    mw  = df.groupby("bioguide_id")["match_weight"].first().reset_index()
    agg = agg.merge(mw, on="bioguide_id", how="left")

    counts = agg.groupby("bioguide_id")["aligned_period"].nunique()
    both   = counts[counts == 2].index
    agg    = agg[agg["bioguide_id"].isin(both)].copy()

    agg["post"]         = (agg["aligned_period"] == "post").astype(int)
    agg["treat"]        = (agg["group"] == "treatment").astype(int)
    agg["treat_x_post"] = agg["treat"] * agg["post"]
    agg["match_weight"] = pd.to_numeric(agg["match_weight"],
                                        errors="coerce").fillna(1.0)

    conf = (matched[["bioguide_id", "party", "dw_nom_dim1_baseline",
                     "house_tenure_at_event", "cohort"]]
            .drop_duplicates("bioguide_id").copy())
    conf["party_D"]       = (conf["party"] == "D").astype(float)
    conf["ideology"]      = conf["dw_nom_dim1_baseline"]
    conf["house_tenure"]  = conf["house_tenure_at_event"]
    conf["cohort_cong"]   = pd.to_numeric(conf["cohort"], errors="coerce")
    conf["ideol_extreme"] = conf["dw_nom_dim1_baseline"].abs()

    agg = agg.merge(conf[["bioguide_id", "party_D", "ideology",
                           "house_tenure", "cohort_cong", "ideol_extreme"]],
                    on="bioguide_id", how="left")
    return agg


# ---------------------------------------------------------------------------
# Part A: All-subset confounder model comparison
# ---------------------------------------------------------------------------

CONFOUNDER_NAMES = {
    "party_D":       "A:party",
    "ideology":      "B:ideology",
    "house_tenure":  "C:tenure",
    "cohort_cong":   "D:cohort",
    "ideol_extreme": "E:|ideo|",
}
COLS  = list(CONFOUNDER_NAMES.keys())
BASE  = "Y_bar ~ treat + post + treat_x_post"


def sig(p):
    """Convert a p-value to an APA significance star string."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"


def fit_subset(agg, subset_cols, label):
    """Fit one model from the all-subset grid and return key statistics as a dict."""
    formula = BASE + ((" + " + " + ".join(subset_cols)) if subset_cols else "")
    d = agg.dropna(subset=["Y_bar", "match_weight"])
    res = smf.wls(formula, data=d, weights=d["match_weight"]).fit(cov_type="HC3")
    coef = res.params.get("treat_x_post", np.nan)
    se   = res.bse.get("treat_x_post",    np.nan)
    p    = res.pvalues.get("treat_x_post", np.nan)
    ci   = res.conf_int()
    k    = int(res.df_model)      # number of predictors (excl. intercept)
    return {
        "model":      label,
        "n_conf":     len(subset_cols),
        "confounders": "+".join([CONFOUNDER_NAMES[c] for c in subset_cols]) if subset_cols else "none",
        "n_obs":      int(res.nobs),
        "k_params":   k,
        "coef":       round(coef, 6),
        "se":         round(se,   6),
        "ci_lo":      round(ci.loc["treat_x_post", 0], 6) if "treat_x_post" in ci.index else np.nan,
        "ci_hi":      round(ci.loc["treat_x_post", 1], 6) if "treat_x_post" in ci.index else np.nan,
        "p":          round(p, 4),
        "sig":        sig(p),
        "aic":        round(res.aic, 4),
        "bic":        round(res.bic, 4),
        "r2_adj":     round(res.rsquared_adj, 4),
    }


def part_a_all_subsets(agg):
    """Fit all 2^5 = 32 confounder combinations, rank by AIC, print summary, and return results."""
    rows = []
    for k in range(len(COLS) + 1):
        for combo in itertools.combinations(COLS, k):
            label = f"M({k}): " + ("+".join([CONFOUNDER_NAMES[c] for c in combo])
                                   if combo else "baseline")
            rows.append(fit_subset(agg, list(combo), label))

    results = pd.DataFrame(rows).sort_values("aic")
    results["delta_aic"] = results["aic"] - results["aic"].iloc[0]  # distance from best AIC
    results["delta_bic"] = results["bic"] - results["bic"].min()

    results.to_csv(OUT_DIR / "13_all_subset_model_comparison.csv", index=False)
    print(f"Saved: 13_all_subset_model_comparison.csv")

    # --- summary table (key results) ---
    print(f"\n{'Rank':>4}  {'Model':55s}  {'coef':>9}  {'p':>6}  {'sig':>3}  {'AIC':>10}  {'ΔAIC':>7}  {'BIC':>10}  {'ΔBIC':>7}")
    print("-" * 120)
    for rank, (_, r) in enumerate(results.head(15).iterrows(), 1):
        print(f"{rank:4d}  {r['model']:55s}  {r['coef']:+9.6f}  {r['p']:6.4f}  "
              f"{r['sig']:>3}  {r['aic']:10.2f}  {r['delta_aic']:7.2f}  "
              f"{r['bic']:10.2f}  {r['delta_bic']:7.2f}")

    return results


# ---------------------------------------------------------------------------
# Part B: LMM vs OLS (word-level)
# ---------------------------------------------------------------------------

def part_b_lmm(df):
    """Compare OLS M5 (speaker+word FE) vs LMM (speaker random intercept), save comparison CSV."""
    try:
        import statsmodels.formula.api as smf_m
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        print("WARNING: statsmodels MixedLM not available — skipping LMM comparison.")
        return None

    df = df.copy()
    df["match_weight"] = pd.to_numeric(df["match_weight"], errors="coerce").fillna(1.0)
    w = df["match_weight"]

    # OLS M5: speaker + word fixed effects, cluster SE by speaker
    m5 = smf.wls(
        "Y ~ post + treat_x_post + C(bioguide_id) + C(target_word)",
        data=df, weights=w
    ).fit(cov_type="cluster", cov_kwds={"groups": df["bioguide_id"]})

    coef_m5 = m5.params.get("treat_x_post", np.nan)
    se_m5   = m5.bse.get("treat_x_post",    np.nan)
    p_m5    = m5.pvalues.get("treat_x_post", np.nan)

    # LMM: speaker as random intercept, word as fixed effect, REML estimation
    try:
        lmm = smf.mixedlm(
            "Y ~ post + treat_x_post + C(target_word)",
            data=df,
            groups=df["bioguide_id"]
        ).fit(reml=True, method="lbfgs")

        coef_lmm = lmm.params.get("treat_x_post", np.nan)
        se_lmm   = lmm.bse.get("treat_x_post",    np.nan)
        p_lmm    = lmm.pvalues.get("treat_x_post", np.nan)

        # ICC = speaker variance / (speaker variance + residual variance)
        icc = lmm.cov_re.iloc[0,0] / (lmm.cov_re.iloc[0,0] + lmm.scale)

        comparison = pd.DataFrame([
            {"model": "OLS M5 (Speaker FE + Word FE, cluster SE)",
             "coef": round(coef_m5, 6), "se": round(se_m5, 6),
             "p": round(p_m5, 4), "sig": sig(p_m5),
             "aic": round(m5.aic, 2), "n_obs": int(m5.nobs)},
            {"model": "LMM (Speaker random + Word FE, REML)",
             "coef": round(coef_lmm, 6), "se": round(se_lmm, 6),
             "p": round(p_lmm, 4), "sig": sig(p_lmm),
             "aic": np.nan, "n_obs": int(lmm.nobs)},
        ])
        comparison.to_csv(OUT_DIR / "14_lmm_vs_ols_comparison.csv", index=False)
        print(f"Saved: 14_lmm_vs_ols_comparison.csv")
        return comparison

    except Exception as e:
        print(f"WARNING: LMM fitting failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Figure 13: All-subset AIC landscape
# ---------------------------------------------------------------------------

def fig13_all_subsets(results):
    """Plot AIC landscape: scatter by n_conf and ΔAIC bar for top 15 models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: AIC by number of confounders (jitter)
    ax = axes[0]
    for k in range(6):
        sub = results[results["n_conf"] == k]
        jitter = np.random.default_rng(k).uniform(-0.15, 0.15, len(sub))
        c = "#E05C5C" if k == results.iloc[0]["n_conf"] else "#5C8AE0"
        ax.scatter(k + jitter, sub["aic"], alpha=0.6, s=40, color=c)

    # Mark best model
    best = results.iloc[0]
    ax.scatter(best["n_conf"], best["aic"], color="black", s=120,
               marker="*", zorder=5, label=f"Best: {best['confounders']}")
    ax.set_xlabel("Number of confounders in model", fontsize=11)
    ax.set_ylabel("AIC", fontsize=11)
    ax.set_title("A. AIC by Number of Confounders\n(32 models, each dot = one combination)",
                 fontsize=10)
    ax.legend(fontsize=8)

    # Right: ΔAIC for top 15 models
    ax2 = axes[1]
    top15 = results.head(15).iloc[::-1]
    colors = ["#E05C5C" if i == 14 else "#5C8AE0" for i in range(15)]
    ax2.barh(range(15), top15["delta_aic"], color=colors, alpha=0.8, height=0.7)
    ax2.set_yticks(range(15))
    ax2.set_yticklabels(
        [r["confounders"] if r["confounders"] != "none" else "baseline"
         for _, r in top15.iterrows()],
        fontsize=8)
    ax2.axvline(2, color="gray", linewidth=0.8, linestyle="--",
                label="ΔAIC = 2 (substantial difference)")
    ax2.set_xlabel("ΔAIC from best model", fontsize=11)
    ax2.set_title("B. Top 15 Models by AIC\n(red = best model)", fontsize=10)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "fig10_all_subsets.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run Part A (all-subset model comparison) and Part B (LMM vs OLS); save outputs."""
    print("Loading data …")
    df      = pd.read_csv(OUT_DIR / "00_did_analysis_data.csv")
    matched = pd.read_csv(REPO / "03_output/phase2_matching/matched_sample.csv",
                          dtype={"bioguide_id": str})
    agg     = build_speaker_df(df, matched)

    results = part_a_all_subsets(agg)
    fig13_all_subsets(results)

    part_b_lmm(df)


if __name__ == "__main__":
    main()
