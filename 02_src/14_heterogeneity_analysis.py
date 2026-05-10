"""
Layer 3: Heterogeneity analysis.

1. By party (Democrat / Republican)
2. By cohort (early 107-109 vs late 110-114 transitions)
3. By chamber origin (all treatment are H→S, but check within-treatment variation)
4. Visualization: subgroup DiD coefficients
"""

from pathlib import Path
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
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def aggregate_speaker_period(df, groupby_extra=None):
    """Weighted mean Y per speaker × period; keeps only speakers with both pre and post."""
    def wmean(g):
        w = g["occurrence_count"].clip(lower=1)
        return np.average(g["Y"], weights=w)

    keys = ["bioguide_id", "speaker", "group", "aligned_period"]
    if groupby_extra:
        keys += groupby_extra

    agg = (df.groupby(keys, dropna=False)
             .apply(wmean)
             .reset_index(name="Y_bar"))

    mw = df.groupby("bioguide_id")["match_weight"].first().reset_index()
    agg = agg.merge(mw, on="bioguide_id", how="left")

    counts = agg.groupby("bioguide_id")["aligned_period"].nunique()
    both   = counts[counts == 2].index
    agg    = agg[agg["bioguide_id"].isin(both)].copy()

    agg["post"]         = (agg["aligned_period"] == "post").astype(int)
    agg["treat"]        = (agg["group"] == "treatment").astype(int)
    agg["treat_x_post"] = agg["treat"] * agg["post"]
    agg["match_weight"] = pd.to_numeric(agg["match_weight"],
                                        errors="coerce").fillna(1.0)
    return agg


def fit_did(df, label="", cov="HC3"):
    """Fit speaker-level DiD (WLS, no FE) and return a result dict, or None if insufficient data."""
    d = df.dropna(subset=["Y_bar", "match_weight"])
    if len(d) < 8 or d["treat"].nunique() < 2:
        return None
    res = smf.wls("Y_bar ~ post + treat_x_post",
                  data=d, weights=d["match_weight"]).fit(cov_type=cov)
    coef = res.params.get("treat_x_post", np.nan)
    se   = res.bse.get("treat_x_post",    np.nan)
    pval = res.pvalues.get("treat_x_post", np.nan)
    ci   = res.conf_int()
    n_t  = int((d[d["post"]==0]["treat"]==1).sum())
    n_c  = int((d[d["post"]==0]["treat"]==0).sum())
    return {
        "label":   label,
        "n_treat": n_t,
        "n_ctrl":  n_c,
        "n_obs":   int(res.nobs),
        "coef":    round(coef, 6),
        "se":      round(se,   6),
        "ci_lo":   round(ci.loc["treat_x_post", 0], 6) if "treat_x_post" in ci.index else np.nan,
        "ci_hi":   round(ci.loc["treat_x_post", 1], 6) if "treat_x_post" in ci.index else np.nan,
        "p":       round(pval, 4),
        "sig":     "***" if pval<0.01 else "**" if pval<0.05 else "*" if pval<0.1 else "ns",
    }


def randomization_inference(agg, n_perm=4999, seed=42):
    """Permute treatment labels and return two-sided RI p-value for the DiD statistic."""
    rng = np.random.default_rng(seed)
    spk = agg[["bioguide_id","treat","match_weight"]].drop_duplicates("bioguide_id")
    pre  = agg[agg["post"]==0][["bioguide_id","Y_bar"]].rename(columns={"Y_bar":"Y_pre"})
    post = agg[agg["post"]==1][["bioguide_id","Y_bar"]].rename(columns={"Y_bar":"Y_post"})
    spk  = spk.merge(pre,on="bioguide_id").merge(post,on="bioguide_id")
    spk["delta"] = spk["Y_post"] - spk["Y_pre"]

    def stat(df):
        wt = df[df["treat"]==1]["match_weight"]
        wc = df[df["treat"]==0]["match_weight"]
        if wt.sum()==0 or wc.sum()==0: return np.nan
        return (np.average(df[df["treat"]==1]["delta"], weights=wt) -
                np.average(df[df["treat"]==0]["delta"], weights=wc))

    obs = stat(spk)
    n_t = int(spk["treat"].sum())
    perms = [stat(spk.assign(treat=lambda d: (
        np.isin(np.arange(len(d)), rng.choice(len(d), n_t, replace=False))
        .astype(int)))) for _ in range(n_perm)]
    perms = np.array([p for p in perms if not np.isnan(p)])
    return round(float(np.mean(np.abs(perms) >= np.abs(obs))), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run heterogeneity subgroup DiD analyses, save results, and generate figure 5."""
    print("Loading DiD analysis data …")
    df = pd.read_csv(OUT_DIR / "00_did_analysis_data.csv")
    matched = pd.read_csv(
        REPO / "03_output/phase2_matching/matched_sample.csv",
        dtype={"bioguide_id": str})

    # Derive cohort group from pseudo_event_congress (<=109 = early transition era)
    treated = matched[matched["treated_or_control"]=="treated"][
        ["bioguide_id","cohort","pseudo_event_congress"]].copy()
    treated["pseudo_event_congress"] = pd.to_numeric(
        treated["pseudo_event_congress"], errors="coerce")
    treated["cohort_group"] = treated["pseudo_event_congress"].apply(
        lambda x: "Early (≤109)" if x <= 109 else "Late (≥110)")

    df = df.merge(treated[["bioguide_id","cohort_group"]],
                  on="bioguide_id", how="left")

    results = []

    # --- baseline (full sample) ---
    agg_all = aggregate_speaker_period(df)
    row = fit_did(agg_all, "Full sample")
    p_ri = randomization_inference(agg_all)
    row["p_ri"] = p_ri
    results.append(row)

    # --- heterogeneity by party ---
    treat_party = (matched[matched["treated_or_control"]=="treated"]
                   [["bioguide_id","party"]].copy())

    for party in ["D", "R"]:
        bios_treat = set(treat_party[treat_party["party"]==party]["bioguide_id"])
        # Subset: treatment speakers of that party + all controls
        df_sub = df[(df["group"]=="control") |
                    (df["bioguide_id"].isin(bios_treat))].copy()
        agg = aggregate_speaker_period(df_sub)
        row = fit_did(agg, f"Party={party}")
        if row:
            p_ri = randomization_inference(agg, n_perm=4999)
            row["p_ri"] = p_ri
            results.append(row)

    # Interaction model: party × treat_x_post (test whether D vs R effect differs)
    agg_full = agg_all.merge(
        treat_party.rename(columns={"party":"t_party"}),
        on="bioguide_id", how="left")
    agg_full["is_dem"] = (agg_full["t_party"]=="D").astype(float)
    ctrl_party = (df[df["group"]=="control"]
                  .groupby("bioguide_id")["party"].first()
                  .map({"D":1.0,"R":0.0})
                  .rename("ctrl_party"))
    agg_full = agg_full.merge(ctrl_party, on="bioguide_id", how="left")
    agg_full["is_dem"] = agg_full["is_dem"].fillna(agg_full["ctrl_party"])

    try:
        res_int = smf.wls(
            "Y_bar ~ post + treat_x_post + is_dem + treat_x_post:is_dem",
            data=agg_full.dropna(subset=["Y_bar","is_dem"]),
            weights=agg_full.dropna(subset=["Y_bar","is_dem"])["match_weight"]
        ).fit(cov_type="HC3")
        int_p = res_int.pvalues.get("treat_x_post:is_dem", np.nan)
        if int_p < 0.1:
            int_coef = res_int.params.get("treat_x_post:is_dem", np.nan)
            print(f"WARNING: significant party interaction (coef={int_coef:+.5f}, p={int_p:.4f})")
    except Exception as e:
        print(f"WARNING: party interaction test failed: {e}")

    # --- heterogeneity by cohort ---
    cohort_map = treated.set_index("bioguide_id")["cohort_group"]

    for cgroup in ["Early (≤109)", "Late (≥110)"]:
        bios = set(cohort_map[cohort_map==cgroup].index)
        df_sub = df[(df["group"]=="control") |
                    (df["bioguide_id"].isin(bios))].copy()
        agg = aggregate_speaker_period(df_sub)
        row = fit_did(agg, f"Cohort={cgroup}")
        if row:
            p_ri = randomization_inference(agg, n_perm=4999)
            row["p_ri"] = p_ri
            results.append(row)

    # --- party × cohort cross-tabs ---
    treat_meta = (treated.merge(treat_party, on="bioguide_id", how="left")
                  [["bioguide_id","party","cohort_group"]])

    for (party, cgroup), sub in treat_meta.groupby(["party","cohort_group"]):
        bios = set(sub["bioguide_id"])
        df_sub = df[(df["group"]=="control") |
                    (df["bioguide_id"].isin(bios))].copy()
        agg = aggregate_speaker_period(df_sub)
        row = fit_did(agg, f"Party={party} × {cgroup}")
        if row and row["n_treat"] >= 3:
            results.append(row)

    # --- save results ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "10_heterogeneity_results.csv", index=False)
    print(f"Saved: {OUT_DIR}/10_heterogeneity_results.csv")

    # --- figure 5: subgroup DiD forest plot ---
    fig, ax = plt.subplots(figsize=(7, 5))

    plot_rows = results_df[results_df["label"].isin([
        "Full sample",
        "Party=D", "Party=R",
        "Cohort=Early (≤109)", "Cohort=Late (≥110)"
    ])].copy()

    labels_short = {
        "Full sample": "Full sample",
        "Party=D": "Democrat",
        "Party=R": "Republican",
        "Cohort=Early (≤109)": "Early cohort (≤109)",
        "Cohort=Late (≥110)": "Late cohort (≥110)",
    }
    plot_rows["label_short"] = plot_rows["label"].map(labels_short)
    plot_rows = plot_rows.iloc[::-1]

    colors = ["#444444","#3070B0","#C0392B","#27AE60","#8E44AD"]
    for i, (_, row) in enumerate(plot_rows.iterrows()):
        color = colors[i % len(colors)]
        ax.plot(row["coef"], i, "o", color=color, markersize=8, zorder=3)
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], "-",
                color=color, linewidth=2, zorder=2)
        sig_label = f"p={row['p']:.3f}{row['sig']}"
        ax.text(row["ci_hi"]+0.001, i, sig_label,
                va="center", fontsize=8.5, color=color)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(plot_rows)))
    ax.set_yticklabels(plot_rows["label_short"].tolist(), fontsize=10)
    ax.set_xlabel("DiD Coefficient (cosine similarity to frame centroid)", fontsize=10)
    ax.set_title("Heterogeneity Analysis: Subgroup DiD Estimates", fontsize=11)

    # Visual separator between full-sample row and subgroup rows
    ax.axhline(3.5, color="gray", linewidth=0.5, linestyle=":")
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] < -0.01 else -0.012,
            4.7, "Party", fontsize=8, color="gray", style="italic")
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] < -0.01 else -0.012,
            1.7, "Cohort", fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    path = FIG_DIR / "fig05_heterogeneity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # --- heterogeneity summary table ---
    print("\n=== HETEROGENEITY SUMMARY ===")
    cols = ["label","n_treat","n_ctrl","coef","se","p","sig","p_ri"]
    display = results_df[results_df["label"].isin(
        ["Full sample","Party=D","Party=R",
         "Cohort=Early (≤109)","Cohort=Late (≥110)"])]
    print(display[cols].to_string(index=False))


if __name__ == "__main__":
    main()
