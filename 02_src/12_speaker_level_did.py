"""
Path B: Speaker-level DiD.

Aggregates word-period observations to speaker × period level.
True unit of analysis = speaker. N ≈ 25 treatment + control speakers.

Steps:
  1. Load did_analysis_data.csv (word-level Y already computed)
  2. Aggregate to speaker × period: weighted mean Y (weight = occurrence_count)
  3. Keep only speakers with BOTH pre and post observations
  4. DiD: Y_bar ~ post + treat_x_post, HC2/HC3 robust SE
  5. Also run frame-level speaker aggregation
  6. Randomization Inference (Path A) on speaker-level data
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import t as t_dist

REPO    = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "03_output" / "did_results"
DATA    = OUT_DIR / "00_did_analysis_data.csv"


# ---------------------------------------------------------------------------
# Step 1: Aggregate to speaker × period
# ---------------------------------------------------------------------------

def aggregate_speaker_period(df: pd.DataFrame,
                              frame: str | None = None) -> pd.DataFrame:
    """
    Weighted mean Y per speaker × period.
    weight = occurrence_count (more occurrences = more reliable embedding).
    Only retain speakers with both pre and post rows.
    """
    if frame:
        df = df[df["frame"] == frame]

    # weighted mean Y per speaker × period
    # groupby only on identity keys — exclude match_weight/cohort to avoid
    # NaN-key dropping (treatment speakers have cohort=NaN)
    def wmean(g):
        w = g["occurrence_count"].clip(lower=1)
        return np.average(g["Y"], weights=w)

    agg = (df.groupby(["bioguide_id", "speaker", "group", "aligned_period"],
                      dropna=False)
             .apply(wmean)
             .reset_index(name="Y_bar"))

    # Attach one match_weight per speaker (take first non-null value)
    mw = (df.groupby("bioguide_id")["match_weight"]
            .first()
            .reset_index())
    agg = agg.merge(mw, on="bioguide_id", how="left")

    # Keep speakers with both pre and post
    counts = agg.groupby("bioguide_id")["aligned_period"].nunique()
    both   = counts[counts == 2].index
    agg    = agg[agg["bioguide_id"].isin(both)].copy()

    # DiD dummies
    agg["post"]         = (agg["aligned_period"] == "post").astype(int)
    agg["treat"]        = (agg["group"] == "treatment").astype(int)
    agg["treat_x_post"] = agg["treat"] * agg["post"]
    agg["match_weight"] = pd.to_numeric(agg["match_weight"],
                                        errors="coerce").fillna(1.0)
    return agg


# ---------------------------------------------------------------------------
# Step 2: Fit speaker-level DiD (HC2 and HC3)
# ---------------------------------------------------------------------------

def fit_speaker_did(agg: pd.DataFrame, cov_type: str = "HC3"):
    """OLS DiD on speaker-level data, no FE (N is small)."""
    d = agg.dropna(subset=["Y_bar"])
    if len(d) < 10:
        return None
    res = smf.wls(
        "Y_bar ~ post + treat_x_post",
        data=d, weights=d["match_weight"]
    ).fit(cov_type=cov_type)
    return res


def summarise(res, label, cov_type):
    """Extract treat_x_post coefficient, CI, and significance from a fitted model."""
    if res is None:
        return None
    coef = res.params.get("treat_x_post", np.nan)
    se   = res.bse.get("treat_x_post",    np.nan)
    pval = res.pvalues.get("treat_x_post", np.nan)
    ci   = res.conf_int()
    return {
        "label":    label,
        "cov_type": cov_type,
        "n_obs":    int(res.nobs),
        "n_treat":  0,   # filled below
        "n_ctrl":   0,
        "coef":     round(coef, 6),
        "se":       round(se,   6),
        "ci_lo":    round(ci.loc["treat_x_post", 0], 6) if "treat_x_post" in ci.index else np.nan,
        "ci_hi":    round(ci.loc["treat_x_post", 1], 6) if "treat_x_post" in ci.index else np.nan,
        "p":        round(pval, 4),
        "sig":      "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "ns")),
    }


# ---------------------------------------------------------------------------
# Path A: Randomization Inference
# ---------------------------------------------------------------------------

def randomization_inference(agg: pd.DataFrame, n_perm: int = 9999,
                             seed: int = 42) -> dict:
    """
    Permute treatment assignment across speakers (within cohort if possible).
    Compute DiD coefficient for each permutation.
    p-value = proportion >= |observed|.

    MacKinnon & Webb (2018): RI is the gold standard for DiD with few
    treated clusters.
    """
    rng = np.random.default_rng(seed)

    # Pivot to speaker-level (one row per speaker)
    spk = agg[["bioguide_id", "treat", "match_weight"]].drop_duplicates("bioguide_id")
    pre  = agg[agg["post"] == 0][["bioguide_id", "Y_bar"]].rename(columns={"Y_bar": "Y_pre"})
    post = agg[agg["post"] == 1][["bioguide_id", "Y_bar"]].rename(columns={"Y_bar": "Y_post"})
    spk  = spk.merge(pre, on="bioguide_id").merge(post, on="bioguide_id")
    spk["delta_Y"] = spk["Y_post"] - spk["Y_pre"]   # first-difference

    # Observed DiD = weighted mean Δ(treat) − weighted mean Δ(ctrl)
    def did_stat(df):
        w_t = df[df["treat"] == 1]["match_weight"]
        w_c = df[df["treat"] == 0]["match_weight"]
        d_t = df[df["treat"] == 1]["delta_Y"]
        d_c = df[df["treat"] == 0]["delta_Y"]
        if w_t.sum() == 0 or w_c.sum() == 0:
            return np.nan
        return np.average(d_t, weights=w_t) - np.average(d_c, weights=w_c)

    observed = did_stat(spk)

    # Permute: reassign treatment labels across speakers
    # (preserving total number of treated)
    n_treated = int(spk["treat"].sum())
    perm_stats = []
    for _ in range(n_perm):
        perm = spk.copy()
        perm["treat"] = 0
        idx = rng.choice(len(perm), size=n_treated, replace=False)
        perm.iloc[idx, perm.columns.get_loc("treat")] = 1
        perm_stats.append(did_stat(perm))

    perm_stats = np.array(perm_stats)
    p_two_sided = np.mean(np.abs(perm_stats) >= np.abs(observed))

    return {
        "observed_did": round(float(observed), 6),
        "perm_mean":    round(float(np.nanmean(perm_stats)), 6),
        "perm_sd":      round(float(np.nanstd(perm_stats)),  6),
        "p_ri":         round(float(p_two_sided), 4),
        "n_perm":       n_perm,
        "n_treated":    n_treated,
        "n_control":    len(spk) - n_treated,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Aggregate to speaker level, run DiD and RI, save results and final summary."""
    print("Loading word-level DiD data …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)

    # --- aggregate to speaker level ---
    agg = aggregate_speaker_period(df)
    n_treat = (agg[agg["post"]==0]["treat"]==1).sum()
    n_ctrl  = (agg[agg["post"]==0]["treat"]==0).sum()

    # --- pooled speaker-level DiD (HC2 and HC3) ---
    results = []
    for cov in ["HC2", "HC3"]:
        res  = fit_speaker_did(agg, cov)
        row  = summarise(res, f"speaker_pooled_{cov}", cov)
        if row:
            row["n_treat"] = n_treat
            row["n_ctrl"]  = n_ctrl
            results.append(row)

    # --- per-frame speaker-level DiD (HC3) ---
    frame_rows = []
    for frame in sorted(df["frame"].unique()):
        agg_f  = aggregate_speaker_period(df, frame=frame)
        if len(agg_f) < 8:
            continue
        res_f  = fit_speaker_did(agg_f, "HC3")
        row_f  = summarise(res_f, frame, "HC3")
        if row_f:
            nt = int((agg_f[agg_f["post"]==0]["treat"]==1).sum())
            nc = int((agg_f[agg_f["post"]==0]["treat"]==0).sum())
            row_f["n_treat"] = nt
            row_f["n_ctrl"]  = nc
            frame_rows.append(row_f)

    # --- randomization inference (pooled and per frame) ---
    ri = randomization_inference(agg, n_perm=9999)
    ri_rows = []
    for frame in sorted(df["frame"].unique()):
        agg_f = aggregate_speaker_period(df, frame=frame)
        if len(agg_f) < 8:
            continue
        ri_f = randomization_inference(agg_f, n_perm=9999)
        ri_rows.append({"frame": frame, **ri_f})

    # --- save outputs ---
    all_results = pd.DataFrame(results + frame_rows)
    all_results.to_csv(OUT_DIR / "03_speaker_level_did.csv", index=False)

    ri_df = pd.DataFrame([{"scope": "pooled", **ri}] +
                         [{"scope": r["frame"], **{k:v for k,v in r.items() if k!="frame"}}
                          for r in ri_rows])
    ri_df.to_csv(OUT_DIR / "05_randomization_inference.csv", index=False)

    print(f"Saved: {OUT_DIR}/03_speaker_level_did.csv")
    print(f"Saved: {OUT_DIR}/05_randomization_inference.csv")

    # --- final summary table (key results) ---
    print("\n=== FINAL SUMMARY ===")
    print("\nSpeaker-level DiD (HC3):")
    cols = ["label","n_obs","n_treat","n_ctrl","coef","se","p","sig"]
    print(all_results[cols].to_string(index=False))
    print(f"\nRandomization Inference (pooled): p = {ri['p_ri']:.4f}")


if __name__ == "__main__":
    main()
