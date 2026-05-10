"""
Finalize all outputs: fix data quality issues, save missing CSVs,
generate missing figures.

Missing CSVs:
  - fdr_correction_results.csv
  - parallel_trends_stats.csv
  - event_study_data.csv
  - descriptive_stats.csv

Data fixes:
  - did_regression_results.csv: fill NaN sig with 'ns'

Missing figures:
  - fig7_robustness.png
  - fig9_ttest_boxplot.png
  - fig10_fdr.png
  - fig11_loo.png
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind, ttest_1samp
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

REPO    = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "03_output" / "did_results"
FIG_DIR = REPO / "03_output" / "figures"
EMB_DIR = REPO / "01_data" / "04_Embeddings"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sig_label(p):
    """Convert a p-value to an APA significance star string."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"


def wmean_agg(df):
    """Aggregate word-level data to speaker × period using occurrence-count-weighted mean Y."""
    def wmean(g):
        """Occurrence-count-weighted mean Y for one group."""
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
    agg["match_weight"] = pd.to_numeric(agg["match_weight"], errors="coerce").fillna(1.0)
    return agg


# ---------------------------------------------------------------------------
# Fix 1: did_regression_results.csv NaN sig
# ---------------------------------------------------------------------------

def fix_regression_results():
    """Fill NaN sig values in did_regression_results.csv with the appropriate star string."""
    path = OUT_DIR / "02_did_regression_results.csv"
    df = pd.read_csv(path)
    df["sig"] = df.apply(
        lambda r: sig_label(r["p_value"]) if pd.isna(r["sig"]) else r["sig"], axis=1)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Save 1: descriptive_stats.csv
# ---------------------------------------------------------------------------

def save_descriptive_stats(df, agg):
    """Compute word- and speaker-level descriptive stats and save to CSV."""
    # Word-level
    word_desc = df.groupby(["group", "aligned_period"])["Y"].agg(
        mean="mean", sd="std", median="median", n="count").reset_index()
    word_desc.columns = ["group", "period", "mean_Y", "sd_Y", "median_Y", "n_obs"]

    # Speaker-level
    spk_desc = agg.groupby(["group", "aligned_period"])["Y_bar"].agg(
        mean="mean", sd="std", median="median", n="count").reset_index()
    spk_desc.columns = ["group", "period", "mean_Ybar", "sd_Ybar", "median_Ybar", "n_speakers"]

    out = pd.merge(word_desc, spk_desc, on=["group", "period"])
    out.to_csv(OUT_DIR / "01_descriptive_stats.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Save 2: parallel_trends_stats.csv
# ---------------------------------------------------------------------------

def save_parallel_trends(df, matched):
    """Recompute congress-level Y, run pre-trend tests, and save parallel-trends stats CSV."""
    def parse_congs(raw):
        if pd.isna(raw): return set()
        return {str(int(x)) for x in str(raw).split(";") if x.strip()}

    def l2_norm(mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1.0, norms)

    # Recompute centroids
    meta_al = pd.read_csv(EMB_DIR / "aligned_metadata.csv")
    emb_al  = l2_norm(np.load(EMB_DIR / "aligned_embeddings.npy").astype(np.float32))
    mask    = (meta_al["group"] == "control") & (meta_al["aligned_period"] == "pre")
    centroids = {}
    for frame, grp in meta_al[mask].reset_index(drop=True).groupby("frame"):
        c = emb_al[mask.values][grp.index.values].mean(axis=0)
        n = np.linalg.norm(c)
        centroids[frame] = c / n if n > 0 else c

    # Congress-level Y for treatment
    meta_raw = pd.read_csv(EMB_DIR / "target_word_speaker_period_metadata.csv")
    emb_raw  = l2_norm(np.load(EMB_DIR / "target_word_speaker_period_embeddings.npy").astype(np.float32))
    extras   = {"F000444", "S000033"}
    mask2    = ((meta_raw["frame"] != "Water") &
                (meta_raw["occurrence_count"] >= 3) &
                (~meta_raw["bioguide_id"].isin(extras)))
    meta_c   = meta_raw[mask2].reset_index(drop=True)
    emb_c    = emb_raw[mask2]
    meta_c["Y"] = [float(np.dot(e, centroids[f])) if f in centroids else np.nan
                   for e, f in zip(emb_c, meta_c["frame"])]

    treated = matched[matched["treated_or_control"] == "treated"]
    treat_meta = meta_c[meta_c["group"] == "treatment"].copy()
    treat_meta["congress_num"] = pd.to_numeric(treat_meta["period"], errors="coerce")

    rows = []
    for _, row in treated.iterrows():
        bio     = row["bioguide_id"]
        pre_set = parse_congs(row["pre_period_congresses"])
        if not pre_set: continue
        pseudo  = int(row["pseudo_event_congress"])
        sp = treat_meta[(treat_meta["bioguide_id"] == bio) &
                        (treat_meta["period"].isin(pre_set))]
        if sp.empty: continue
        for cong, grp in sp.groupby("congress_num"):
            rows.append({"bioguide_id": bio, "congress": int(cong),
                         "rel_congress": int(cong) - pseudo,
                         "Y_mean": grp["Y"].mean(), "n_obs": len(grp)})

    treat_pre_df = pd.DataFrame(rows)
    ctrl_pre     = meta_c[(meta_c["group"] == "control") &
                          (meta_c["period"] == "pre")].groupby("bioguide_id")["Y"].mean()

    # T-test: baseline level
    t_means = treat_pre_df.groupby("bioguide_id")["Y_mean"].mean()
    t_stat, t_p = ttest_ind(t_means, ctrl_pre, equal_var=False)

    # Slope test
    slope_p = np.nan
    slope   = np.nan
    if len(treat_pre_df) >= 10:
        res = smf.ols("Y_mean ~ rel_congress + C(bioguide_id)",
                      data=treat_pre_df).fit(cov_type="HC3")
        slope   = res.params.get("rel_congress", np.nan)
        slope_p = res.pvalues.get("rel_congress", np.nan)

    stats = pd.DataFrame([{
        "test": "Baseline level t-test (treat_pre vs ctrl_pre)",
        "treat_mean": round(t_means.mean(), 6),
        "ctrl_mean":  round(ctrl_pre.mean(), 6),
        "t_stat":     round(t_stat, 4),
        "p":          round(t_p, 4),
        "sig":        sig_label(t_p),
        "interpretation": "No baseline difference" if t_p >= 0.05 else "Groups differ at pre"},
        {"test": "Within-pre time trend (treatment)",
         "treat_mean": np.nan, "ctrl_mean": np.nan,
         "t_stat": round(slope, 6) if not np.isnan(slope) else np.nan,
         "p": round(slope_p, 4) if not np.isnan(slope_p) else np.nan,
         "sig": sig_label(slope_p) if not np.isnan(slope_p) else "n/a",
         "interpretation": "No significant pre-trend" if np.isnan(slope_p) or slope_p >= 0.1
                           else "Significant pre-trend WARNING"}])
    stats.to_csv(OUT_DIR / "08_parallel_trends_stats.csv", index=False)
    treat_pre_df.to_csv(OUT_DIR / "09_event_study_data.csv", index=False)
    return treat_pre_df, ctrl_pre


# ---------------------------------------------------------------------------
# Save 3: fdr_correction_results.csv
# ---------------------------------------------------------------------------

def save_fdr_results():
    """Apply Benjamini-Hochberg FDR correction across frame and heterogeneity tests."""
    frame_did  = pd.read_csv(OUT_DIR / "03_speaker_level_did.csv")
    hetero     = pd.read_csv(OUT_DIR / "10_heterogeneity_results.csv")
    ri_df      = pd.read_csv(OUT_DIR / "05_randomization_inference.csv")

    frame_rows = frame_did[~frame_did["label"].str.startswith("speaker_pooled")].copy()
    hetero_sub = hetero[hetero["label"] != "Full sample"].copy()

    tests = []
    for _, r in frame_rows.iterrows():
        tests.append({"family": "frame_HC3", "label": r["label"],
                      "coef": r["coef"], "p_raw": r["p"]})
    for _, r in hetero_sub.iterrows():
        tests.append({"family": "heterogeneity", "label": r["label"],
                      "coef": r["coef"], "p_raw": r["p"]})

    df_tests = pd.DataFrame(tests)
    # Benjamini-Hochberg FDR correction across all frame + heterogeneity tests jointly
    _, p_fdr, _, _ = multipletests(df_tests["p_raw"], method="fdr_bh", alpha=0.05)
    df_tests["p_fdr"]   = p_fdr.round(4)
    df_tests["fdr_sig"] = (p_fdr < 0.05)
    df_tests["sig_raw"] = df_tests["p_raw"].apply(sig_label)
    df_tests["sig_fdr"] = df_tests["p_fdr"].apply(sig_label)

    # FDR correction for RI p-values (frame-level only; pooled RI excluded)
    ri_frame = ri_df[ri_df["scope"] != "pooled"].copy()
    _, p_fdr_ri, _, _ = multipletests(ri_frame["p_ri"], method="fdr_bh", alpha=0.05)
    ri_frame["p_fdr"]   = p_fdr_ri.round(4)
    ri_frame["fdr_sig"] = (p_fdr_ri < 0.05)
    ri_frame["sig_fdr"] = ri_frame["p_fdr"].apply(sig_label)

    df_tests.to_csv(OUT_DIR / "11_fdr_correction_results.csv", index=False)
    ri_frame.to_csv(OUT_DIR / "12_fdr_ri_results.csv", index=False)
    return df_tests, ri_frame


# ---------------------------------------------------------------------------
# Figure 7: Robustness checks
# ---------------------------------------------------------------------------

def fig7_robustness():
    """Two-panel robustness figure: occurrence thresholds (left) and LOO (right)."""
    rob = pd.read_csv(OUT_DIR / "06_robustness_main.csv")
    loo = pd.read_csv(OUT_DIR / "07_robustness_loo.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: occurrence threshold + drop Economic + wild bootstrap
    ax = axes[0]
    plot_df = rob[rob["label"] != "wild_bootstrap"].copy()
    labels  = ["Baseline\n(occ≥3)", "occ≥2", "occ≥5", "occ≥10", "Drop\nEconomic"]
    colors  = ["#444444"] + ["#5C8AE0"] * 3 + ["#E05C5C"]
    xs = range(len(plot_df))
    bars = ax.bar(xs, plot_df["coef"], color=colors, alpha=0.85, width=0.6)
    ax.errorbar(xs, plot_df["coef"],
                yerr=[plot_df["coef"] - plot_df["ci_lo"],
                      plot_df["ci_hi"] - plot_df["coef"]],
                fmt="none", color="black", capsize=5, linewidth=1.2)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        s = row["sig"] if row["sig"] != "ns" else ""
        ax.text(i, row["ci_hi"] + 0.0003, s, ha="center", fontsize=10, color="black")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("DiD Coefficient", fontsize=10)
    ax.set_title("A. Occurrence Threshold & Frame Sensitivity", fontsize=10)

    # Wild bootstrap annotation
    wb = rob[rob["label"] == "wild_bootstrap"].iloc[0]
    ax.text(0, -0.002, f"Wild bootstrap p = {wb['p_boot']:.3f}",
            fontsize=8, color="gray", style="italic")

    # Right: LOO plot
    ax2 = axes[1]
    loo_t = loo[loo["dropped_bio"].notna()].copy()
    loo_t = loo_t.sort_values("coef")
    base  = rob[rob["label"] == "baseline_occ3"].iloc[0]["coef"]
    colors_loo = ["#E05C5C" if r["p"] >= 0.05 else "#5C8AE0"
                  for _, r in loo_t.iterrows()]
    ax2.barh(range(len(loo_t)), loo_t["coef"], color=colors_loo, alpha=0.7, height=0.7)
    ax2.errorbar(loo_t["coef"], range(len(loo_t)),
                 xerr=[loo_t["coef"] - loo_t["ci_lo"],
                       loo_t["ci_hi"] - loo_t["coef"]],
                 fmt="none", color="black", capsize=2, linewidth=0.8)
    ax2.axvline(0, color="black", linewidth=0.7, linestyle="--")
    ax2.axvline(base, color="gray", linewidth=0.7, linestyle=":",
                label=f"Baseline coef = {base:.5f}")
    ax2.set_yticks(range(len(loo_t)))
    ax2.set_yticklabels(
        [r["dropped_speaker"].split(",")[0] for _, r in loo_t.iterrows()],
        fontsize=6.5)
    ax2.set_xlabel("DiD Coefficient", fontsize=10)
    ax2.set_title("B. Leave-One-Out (by treatment speaker)", fontsize=10)

    sig_patch   = mpatches.Patch(color="#5C8AE0", alpha=0.7, label="p < 0.05")
    insig_patch = mpatches.Patch(color="#E05C5C", alpha=0.7, label="p ≥ 0.05")
    ax2.legend(handles=[sig_patch, insig_patch], fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "fig06_robustness.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 9: T-test boxplot
# ---------------------------------------------------------------------------

def fig9_ttest(df):
    """Boxplot + mean-ΔY bar chart comparing treatment vs control pre→post change."""
    agg   = wmean_agg(df)
    pivot = agg.pivot_table(index=["bioguide_id", "group"],
                            columns="aligned_period", values="Y_bar").reset_index()
    pivot["delta_Y"] = pivot["post"] - pivot["pre"]

    treat = pivot[pivot["group"] == "treatment"]["delta_Y"]
    ctrl  = pivot[pivot["group"] == "control"]["delta_Y"]
    t_stat, p_val = ttest_ind(treat, ctrl, equal_var=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    COLORS = {"treatment": "#E05C5C", "control": "#5C8AE0"}

    # Left: boxplot of delta_Y
    ax = axes[0]
    bp = ax.boxplot([treat, ctrl], patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    widths=0.5)
    for patch, color in zip(bp["boxes"], ["#E05C5C", "#5C8AE0"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.scatter([1] * len(treat), treat, alpha=0.5, color="#E05C5C", s=30, zorder=3)
    ax.scatter([2] * len(ctrl),  ctrl,  alpha=0.3, color="#5C8AE0", s=15, zorder=3)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Treatment\n(n=25)", "Control\n(n=56)"], fontsize=11)
    ax.set_ylabel("ΔY (post − pre)", fontsize=11)
    ax.set_title(f"A. Distribution of Pre→Post Change\n"
                 f"Welch t = {t_stat:.3f},  p = {p_val:.4f} {sig_label(p_val)}",
                 fontsize=10)

    # Right: mean ΔY with CI
    ax2 = axes[1]
    groups = ["Treatment", "Control"]
    means  = [treat.mean(), ctrl.mean()]
    sems   = [treat.sem(), ctrl.sem()]
    bars   = ax2.bar([0, 1], means, color=["#E05C5C", "#5C8AE0"],
                     alpha=0.8, width=0.5)
    ax2.errorbar([0, 1], means, yerr=[1.96 * s for s in sems],
                 fmt="none", color="black", capsize=8, linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(groups, fontsize=11)
    ax2.set_ylabel("Mean ΔY (post − pre)", fontsize=11)
    did_val = treat.mean() - ctrl.mean()
    ax2.set_title(f"B. Mean Change ± 95% CI\nDiD = {did_val:+.5f}", fontsize=10)

    # Bracket for DiD
    y_top = max(means) + 2 * max(sems) + 0.002
    ax2.annotate("", xy=(0, y_top), xytext=(1, y_top),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax2.text(0.5, y_top + 0.001,
             f"DiD={did_val:+.4f}\np={p_val:.3f}{sig_label(p_val)}",
             ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "fig07_ttest_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 10: FDR correction summary
# ---------------------------------------------------------------------------

def fig10_fdr():
    """Scatter plot showing raw vs BH-adjusted p-values for frame-level HC3 and RI tests."""
    fdr = pd.read_csv(OUT_DIR / "11_fdr_correction_results.csv")
    ri  = pd.read_csv(OUT_DIR / "12_fdr_ri_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, title, p_col, fdr_col in [
        (axes[0], fdr[fdr["family"] == "frame_HC3"],
         "A. Frame-level HC3 (n=11)", "p_raw", "p_fdr"),
        (axes[1], ri,
         "B. Frame-level RI (n=11)", "p_ri", "p_fdr"),
    ]:
        data = data.sort_values(p_col)
        n = len(data)
        x = range(n)
        ax.scatter(x, data[p_col],  color="#5C8AE0", s=60, zorder=3, label="Raw p")
        ax.scatter(x, data[fdr_col], color="#E05C5C", marker="D", s=50,
                   zorder=3, label="BH-adjusted p")
        ax.axhline(0.05, color="black", linewidth=0.8, linestyle="--", label="α = 0.05")
        ax.axhline(0.10, color="gray",  linewidth=0.6, linestyle=":",  label="α = 0.10")
        ax.set_xticks(x)
        ax.set_xticklabels(data["scope"].tolist() if "scope" in data.columns
                           else data["label"].tolist(),
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("p-value", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    path = FIG_DIR / "fig08_fdr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 11: LOO detailed plot
# ---------------------------------------------------------------------------

def fig11_loo():
    loo  = pd.read_csv(OUT_DIR / "07_robustness_loo.csv")
    base = pd.read_csv(OUT_DIR / "06_robustness_main.csv")
    base_coef = base[base["label"] == "baseline_occ3"].iloc[0]["coef"]

    loo = loo.sort_values("coef", ascending=False).reset_index(drop=True)
    loo["delta"] = loo["coef"] - base_coef
    loo["speaker_short"] = loo["dropped_speaker"].str.split(",").str[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#E05C5C" if p >= 0.05 else "#5C8AE0" for p in loo["p"]]
    ax.barh(range(len(loo)), loo["coef"], color=colors, alpha=0.75, height=0.7)
    ax.errorbar(loo["coef"], range(len(loo)),
                xerr=[loo["coef"] - loo["ci_lo"], loo["ci_hi"] - loo["coef"]],
                fmt="none", color="black", capsize=3, linewidth=0.8)
    ax.axvline(0,         color="black", linewidth=0.7, linestyle="--")
    ax.axvline(base_coef, color="gray",  linewidth=1.0, linestyle=":",
               label=f"Baseline = {base_coef:.5f}")
    ax.set_yticks(range(len(loo)))
    ax.set_yticklabels(loo["speaker_short"].tolist(), fontsize=8)
    ax.set_xlabel("DiD Coefficient when speaker is dropped", fontsize=10)
    ax.set_title("Leave-One-Out: Treatment Speaker Influence\n"
                 "(red = no longer significant at p<0.05 after drop)", fontsize=10)

    sig_p   = mpatches.Patch(color="#5C8AE0", alpha=0.75, label="p < 0.05 (still significant)")
    insig_p = mpatches.Patch(color="#E05C5C", alpha=0.75, label="p ≥ 0.05 (no longer sig)")
    ax.legend(handles=[sig_p, insig_p, plt.Line2D([0],[0], color="gray",
              linestyle=":", linewidth=1, label=f"Baseline coef")],
              fontsize=8, loc="lower right")

    plt.tight_layout()
    path = FIG_DIR / "fig09_loo.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("Finalizing all outputs")
    print("=" * 55)
    print()

    df      = pd.read_csv(OUT_DIR / "00_did_analysis_data.csv")
    matched = pd.read_csv(
        REPO / "03_output/phase2_matching/matched_sample.csv",
        dtype={"bioguide_id": str})
    agg     = wmean_agg(df)

    print("--- Fixing data quality ---")
    fix_regression_results()
    print()

    print("--- Saving missing CSVs ---")
    save_descriptive_stats(df, agg)
    save_parallel_trends(df, matched)
    save_fdr_results()
    print()

    print("--- Generating missing figures ---")
    fig7_robustness()
    fig9_ttest(df)
    fig10_fdr()
    fig11_loo()
    print()

    print("=" * 55)
    print("All outputs complete. Final inventory:")
    print("=" * 55)
    print()
    print("CSVs:")
    for f in sorted((OUT_DIR).glob("*.csv")):
        print(f"  {f.name:45s} {f.stat().st_size/1024:7.1f} KB")
    print()
    print("Figures:")
    for f in sorted(FIG_DIR.glob("fig*.png")):
        print(f"  {f.name:45s} {f.stat().st_size/1024:7.1f} KB")


if __name__ == "__main__":
    main()
