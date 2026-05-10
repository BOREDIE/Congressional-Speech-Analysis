"""
Layer 2: Parallel trends check + visualization.

1. Parallel trends test
   - For treatment speakers: compute Y at each pre-period congress
     and test if there is a within-pre time trend (slope ≠ 0)
   - Compare treatment vs control pre-period baseline levels (t-test)
   - Event study: plot average Y at each congress relative to transition

2. Visualizations
   - Fig 1: Mean Y_bar by group × period (bar + CI)
   - Fig 2: DiD coefficients by frame (speaker-level, with CI)
   - Fig 3: Event study plot (treatment vs control, relative time)
   - Fig 4: Parallel trends — within-pre Y by congress
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

REPO    = Path(__file__).resolve().parents[1]
EMB_DIR = REPO / "01_data" / "04_Embeddings"
OUT_DIR = REPO / "03_output" / "did_results"
FIG_DIR = REPO / "03_output" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers: recompute Y from raw congress-level embeddings
# ---------------------------------------------------------------------------

def l2_norm(mat):
    """Row-wise L2 normalisation; rows with zero norm are left unchanged."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1.0, norms)


def load_centroids():
    """Re-derive frame centroids from control×pre (same as did_analysis.py)."""
    meta = pd.read_csv(EMB_DIR / "aligned_metadata.csv")
    emb  = np.load(EMB_DIR / "aligned_embeddings.npy").astype(np.float32)
    emb  = l2_norm(emb)
    mask = (meta["group"] == "control") & (meta["aligned_period"] == "pre")
    ref_emb  = emb[mask.values]
    ref_meta = meta[mask].reset_index(drop=True)
    centroids = {}
    for frame, grp in ref_meta.groupby("frame"):
        c = ref_emb[grp.index.values].mean(axis=0)
        n = np.linalg.norm(c)
        centroids[frame] = c / n if n > 0 else c
    return centroids


def compute_Y_congress_level(centroids):
    """
    Load the pre-alignment embeddings (congress-level for treatment),
    compute Y = cosine_sim to frame centroid, return dataframe.
    """
    meta_raw = pd.read_csv(EMB_DIR / "target_word_speaker_period_metadata.csv")
    emb_raw  = np.load(EMB_DIR / "target_word_speaker_period_embeddings.npy").astype(np.float32)
    emb_raw  = l2_norm(emb_raw)

    # Filter: no Water frame, occ >= 3, exclude extras
    extras = {"F000444", "S000033"}
    mask = (
        (meta_raw["frame"] != "Water") &
        (meta_raw["occurrence_count"] >= 3) &
        (~meta_raw["bioguide_id"].isin(extras))
    )
    meta = meta_raw[mask].reset_index(drop=True)
    emb  = emb_raw[mask]

    Y = np.array([
        float(np.dot(row_emb, centroids[frame])) if frame in centroids else np.nan
        for row_emb, frame in zip(emb, meta["frame"])
    ])
    meta = meta.copy()
    meta["Y"] = Y
    return meta


# ---------------------------------------------------------------------------
# 1. Parallel trends
# ---------------------------------------------------------------------------

def parallel_trends_test(meta_cong, matched):
    """
    For each treatment speaker, compute average Y per pre-period congress,
    then test slope within pre period (H0: no pre-trend).
    Also compare treatment vs control pre-period Y level.
    """
    treated = matched[matched["treated_or_control"] == "treated"].copy()

    def parse_congs(raw):
        if pd.isna(raw): return set()
        return {str(int(x)) for x in str(raw).split(";") if x.strip()}

    # Treatment: congress-level Y in pre period
    treat_meta = meta_cong[meta_cong["group"] == "treatment"].copy()
    treat_meta["congress_num"] = pd.to_numeric(treat_meta["period"], errors="coerce")

    rows = []
    for _, row in treated.iterrows():
        bio     = row["bioguide_id"]
        pre_set = parse_congs(row["pre_period_congresses"])
        if not pre_set:
            continue
        sp = treat_meta[
            (treat_meta["bioguide_id"] == bio) &
            (treat_meta["period"].isin(pre_set))
        ]
        if sp.empty:
            continue
        for cong, grp in sp.groupby("congress_num"):
            rows.append({
                "bioguide_id": bio,
                "congress":    int(cong),
                "Y_mean":      grp["Y"].mean(),
                "n_obs":       len(grp),
                # relative congress: 0 = pseudo_event, -1 = one before, etc.
                "rel_congress": int(cong) - int(row["pseudo_event_congress"]),
            })

    treat_pre = pd.DataFrame(rows)

    # Control: pre-period Y (already labeled 'pre')
    ctrl_pre = meta_cong[
        (meta_cong["group"] == "control") &
        (meta_cong["period"] == "pre")
    ].groupby("bioguide_id")["Y"].mean().reset_index(name="Y_mean")

    # Test 1: pre-period baseline level difference (Welch t-test)
    t_pre_mean = treat_pre.groupby("bioguide_id")["Y_mean"].mean()
    c_pre_mean = ctrl_pre["Y_mean"]
    from scipy.stats import ttest_ind
    tstat, pval = ttest_ind(t_pre_mean, c_pre_mean, equal_var=False)
    if pval < 0.05:
        print(f"WARNING: pre-period baseline differs between groups (t={tstat:.3f}, p={pval:.4f})")

    # Test 2: within-pre trend for treatment speakers (H0: slope = 0)
    if len(treat_pre) >= 10:
        res = smf.ols("Y_mean ~ rel_congress + C(bioguide_id)",
                      data=treat_pre).fit(cov_type="HC3")
        slope  = res.params.get("rel_congress", np.nan)
        pslope = res.pvalues.get("rel_congress", np.nan)
        if pslope < 0.1:
            print(f"WARNING: significant within-pre trend (slope={slope:.6f}, p={pslope:.4f})")

    return treat_pre, ctrl_pre


def event_study_data(meta_cong, matched):
    """
    Compute average Y at each relative congress period for treatment speakers.
    rel = congress - pseudo_event_congress.
    """
    treated = matched[matched["treated_or_control"] == "treated"].copy()

    def parse_congs(raw):
        if pd.isna(raw): return set()
        return {str(int(x)) for x in str(raw).split(";") if x.strip()}

    treat_meta = meta_cong[meta_cong["group"] == "treatment"].copy()
    treat_meta["congress_num"] = pd.to_numeric(treat_meta["period"], errors="coerce")

    rows = []
    for _, row in treated.iterrows():
        bio      = row["bioguide_id"]
        pre_set  = parse_congs(row["pre_period_congresses"])
        post_set = parse_congs(row["post_period_congresses"])
        pseudo   = int(row["pseudo_event_congress"])
        sp = treat_meta[treat_meta["bioguide_id"] == bio]
        for cong, grp in sp.groupby("congress_num"):
            period_str = str(int(cong))
            if period_str in pre_set:
                label = "pre"
            elif period_str in post_set:
                label = "post"
            else:
                label = "exclude"
            rows.append({
                "bioguide_id": bio,
                "congress":    int(cong),
                "rel":         int(cong) - pseudo,
                "period":      label,
                "Y_mean":      grp["Y"].mean(),
                "n":           len(grp),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Visualizations
# ---------------------------------------------------------------------------

COLORS = {"treatment": "#E05C5C", "control": "#5C8AE0"}
FRAME_ORDER = ["Exclusion","Migration","Family","Culture","Legality",
               "Sympathetic","Threat","Beneficial","Labor","Crime","Economic"]


def fig1_group_period(did_data_path):
    """Bar chart: mean Y_bar by group × period with CI."""
    df = pd.read_csv(did_data_path)
    df["match_weight"] = pd.to_numeric(df["match_weight"], errors="coerce").fillna(1.0)

    # Aggregate to speaker-period
    def wmean(g):
        w = g["occurrence_count"].clip(lower=1)
        return np.average(g["Y"], weights=w)

    agg = (df.groupby(["bioguide_id","group","aligned_period"])
             .apply(wmean).reset_index(name="Y_bar"))

    summary = agg.groupby(["group","aligned_period"])["Y_bar"].agg(
        mean="mean", sem=lambda x: x.std(ddof=1)/np.sqrt(len(x))
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.array([0, 1])
    width = 0.35
    for i, grp in enumerate(["control", "treatment"]):
        s = summary[summary["group"] == grp].set_index("aligned_period")
        means = [s.loc["pre","mean"], s.loc["post","mean"]]
        sems  = [s.loc["pre","sem"],  s.loc["post","sem"]]
        bars  = ax.bar(x + i*width - width/2, means, width,
                       yerr=sems, capsize=4,
                       color=COLORS[grp], alpha=0.85, label=grp.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels(["Pre-transition", "Post-transition"], fontsize=11)
    ax.set_ylabel("Mean cosine similarity to frame centroid", fontsize=10)
    ax.set_title("Semantic Frame Alignment by Group × Period", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0.77, 0.81)
    plt.tight_layout()
    path = FIG_DIR / "fig01_group_period.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig2_frame_coefs(speaker_did_path):
    """DiD coefficient plot by frame (speaker-level HC3) with CI."""
    df = pd.read_csv(speaker_did_path)
    df = df[~df["label"].str.startswith("speaker_pooled")].copy()
    df = df.rename(columns={"label":"frame"})
    df = df[df["frame"].isin(FRAME_ORDER)].copy()
    df["frame"] = pd.Categorical(df["frame"], categories=FRAME_ORDER, ordered=True)
    df = df.sort_values("coef")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#E05C5C" if c > 0 else "#5C8AE0" for c in df["coef"]]
    ax.barh(df["frame"], df["coef"], xerr=[df["coef"]-df["ci_lo"], df["ci_hi"]-df["coef"]],
            color=colors, alpha=0.8, capsize=4, height=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    for _, row in df.iterrows():
        if row["sig"] not in ("", "ns"):
            ax.text(row["ci_hi"] + 0.001, row["frame"], row["sig"],
                    va="center", fontsize=9, color="#333333")

    ax.set_xlabel("DiD Coefficient (cosine similarity)", fontsize=10)
    ax.set_title("Frame-level DiD Estimates (Speaker-level HC3)", fontsize=11)
    plt.tight_layout()
    path = FIG_DIR / "fig02_frame_coefs.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig3_event_study(ev_df):
    """Event study: average Y by relative congress for treatment speakers."""
    ev = ev_df[ev_df["period"] != "exclude"].copy()
    avg = ev.groupby("rel")["Y_mean"].agg(mean="mean",
          sem=lambda x: x.std(ddof=1)/np.sqrt(len(x))).reset_index()
    avg = avg[avg["rel"].between(-5, 5)]

    fig, ax = plt.subplots(figsize=(7, 4))
    pre  = avg[avg["rel"] < 0]
    post = avg[avg["rel"] > 0]

    for part, color, label in [(pre, "#5C8AE0","Pre (House)"),
                                (post,"#E05C5C","Post (Senate)")]:
        ax.plot(part["rel"], part["mean"], "o-", color=color, label=label)
        ax.fill_between(part["rel"],
                        part["mean"] - 1.96*part["sem"],
                        part["mean"] + 1.96*part["sem"],
                        alpha=0.2, color=color)

    ax.axvline(0, color="gray", linewidth=1, linestyle="--", label="Transition")
    ax.set_xlabel("Congress relative to transition (0 = transition congress)", fontsize=10)
    ax.set_ylabel("Mean Y (cosine similarity)", fontsize=10)
    ax.set_title("Event Study: Semantic Frame Alignment Around H→S Transition", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = FIG_DIR / "fig03_event_study.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig4_parallel_trends(treat_pre, ctrl_pre):
    """Parallel trends: treatment pre-period Y by relative congress."""
    if treat_pre.empty:
        print("  No pre-period data for parallel trends plot.")
        return

    avg = treat_pre.groupby("rel_congress")["Y_mean"].agg(
        mean="mean", sem=lambda x: x.std(ddof=1)/np.sqrt(len(x))
    ).reset_index()
    avg = avg[avg["rel_congress"] < 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(avg["rel_congress"], avg["mean"], "o-", color=COLORS["treatment"],
            label="Treatment (pre-period)")
    ax.fill_between(avg["rel_congress"],
                    avg["mean"] - 1.96*avg["sem"],
                    avg["mean"] + 1.96*avg["sem"],
                    alpha=0.2, color=COLORS["treatment"])

    ctrl_mean = ctrl_pre["Y_mean"].mean()
    ctrl_sem  = ctrl_pre["Y_mean"].sem()
    ax.axhline(ctrl_mean, color=COLORS["control"], linestyle="--",
               label=f"Control pre-period mean ({ctrl_mean:.4f})")
    ax.fill_between(avg["rel_congress"],
                    ctrl_mean - 1.96*ctrl_sem,
                    ctrl_mean + 1.96*ctrl_sem,
                    alpha=0.1, color=COLORS["control"])

    ax.set_xlabel("Congress relative to transition", fontsize=10)
    ax.set_ylabel("Mean Y (cosine similarity)", fontsize=10)
    ax.set_title("Parallel Trends Check: Pre-period Language Trajectory", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = FIG_DIR / "fig04_parallel_trends.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run parallel trends tests and generate all visualization figures."""
    print("Loading centroids and congress-level embeddings …")
    centroids  = load_centroids()
    meta_cong  = compute_Y_congress_level(centroids)
    matched    = pd.read_csv(
        REPO / "03_output/phase2_matching/matched_sample.csv",
        dtype={"bioguide_id": str}
    )

    # --- 1. parallel trends test ---
    treat_pre, ctrl_pre = parallel_trends_test(meta_cong, matched)

    # --- 2. event study data ---
    ev_df = event_study_data(meta_cong, matched)

    # --- 3. generate figures ---
    fig1_group_period(OUT_DIR / "00_did_analysis_data.csv")
    fig2_frame_coefs(OUT_DIR / "03_speaker_level_did.csv")
    fig3_event_study(ev_df)
    fig4_parallel_trends(treat_pre, ctrl_pre)


if __name__ == "__main__":
    main()
