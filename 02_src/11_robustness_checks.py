"""
Robustness checks for the DiD analysis.

1. Occurrence-count thresholds  (occ >= 2, 3, 5, 10)
2. Drop Economic frame
3. Leave-one-out treatment speakers
4. Wild cluster bootstrap (Webb weights, 999 draws)

All checks reuse the same outcome Y and model spec as did_analysis.py:
  Y ~ post + treat_x_post + C(bioguide_id) + C(target_word)
  weights = match_weight, SE clustered by bioguide_id
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

warnings.filterwarnings("ignore")

REPO    = Path(__file__).resolve().parents[1]
EMB_DIR = REPO / "01_data" / "04_Embeddings"
OUT_DIR = REPO / "03_output" / "did_results"

META_PATH = EMB_DIR / "aligned_metadata.csv"
EMB_PATH  = EMB_DIR / "aligned_embeddings.npy"


# ---------------------------------------------------------------------------
# Shared helpers (replicate from did_analysis.py to keep self-contained)
# ---------------------------------------------------------------------------

def l2_norm(mat):
    """Row-wise L2 normalisation; rows with zero norm are left unchanged."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1.0, norms)


def frame_centroids(emb, meta):
    """Compute per-frame L2-normed centroid from control × pre rows only."""
    ref_mask = (meta["group"] == "control") & (meta["aligned_period"] == "pre")
    ref_emb  = emb[ref_mask.values]
    ref_meta = meta[ref_mask].reset_index(drop=True)
    out = {}
    for frame, grp in ref_meta.groupby("frame"):
        c = ref_emb[grp.index.values].mean(axis=0)
        n = np.linalg.norm(c)
        out[frame] = c / n if n > 0 else c
    return out


def add_outcome(meta, emb, centroids):
    """Compute Y (cosine similarity) and DiD dummy columns; return augmented copy."""
    Y = np.array([
        float(np.dot(row_emb, centroids[frame])) if frame in centroids else np.nan
        for row_emb, frame in zip(emb, meta["frame"])
    ])
    meta = meta.copy()
    meta["Y"] = Y
    meta["post"]         = (meta["aligned_period"] == "post").astype(int)
    meta["treat"]        = (meta["group"] == "treatment").astype(int)
    meta["treat_x_post"] = meta["treat"] * meta["post"]
    meta["match_weight"] = pd.to_numeric(meta["match_weight"], errors="coerce").fillna(1.0)
    return meta


def fit_did(df):
    """Fit the main DiD spec (WLS with speaker+word FE, cluster SE); return fitted model or None."""
    d = df.dropna(subset=["Y", "match_weight"])
    if d["bioguide_id"].nunique() < 5 or d["target_word"].nunique() < 2:
        return None
    res = smf.wls(
        "Y ~ post + treat_x_post + C(bioguide_id) + C(target_word)",
        data=d, weights=d["match_weight"]
    ).fit(cov_type="cluster", cov_kwds={"groups": d["bioguide_id"]})
    return res


def extract(res, label):
    """Extract treat_x_post coefficient, SE, CI, and p-value from a fitted model."""
    if res is None:
        return {"label": label, "coef": np.nan, "se": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n": 0}
    coef = res.params.get("treat_x_post", np.nan)
    se   = res.bse.get("treat_x_post",    np.nan)
    pval = res.pvalues.get("treat_x_post", np.nan)
    ci   = res.conf_int()
    return {
        "label": label,
        "coef":  round(coef, 6),
        "se":    round(se,   6),
        "ci_lo": round(ci.loc["treat_x_post", 0], 6) if "treat_x_post" in ci.index else np.nan,
        "ci_hi": round(ci.loc["treat_x_post", 1], 6) if "treat_x_post" in ci.index else np.nan,
        "p":     round(pval, 4),
        "n":     int(res.nobs),
        "sig":   "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "ns")),
    }


# ---------------------------------------------------------------------------
# Wild cluster bootstrap (Webb 6-point weights)
# ---------------------------------------------------------------------------

def wild_cluster_bootstrap(df, n_boot=999, seed=42):
    """
    Wild cluster bootstrap with Webb weights.
    Returns bootstrap p-value for treat_x_post.
    """
    rng      = np.random.default_rng(seed)
    clusters = df["bioguide_id"].unique()
    # Webb 6-point weights
    webb = np.array([-np.sqrt(3/2), -1, -np.sqrt(1/2),
                      np.sqrt(1/2),  1,  np.sqrt(3/2)])

    base_res = fit_did(df)
    if base_res is None:
        return np.nan
    t_obs = base_res.tvalues.get("treat_x_post", np.nan)
    if np.isnan(t_obs):
        return np.nan

    d = df.dropna(subset=["Y", "match_weight"]).copy()
    residuals = base_res.resid

    # Map residuals back to df index
    resid_series = pd.Series(residuals, index=d.index)

    t_boot = []
    for _ in range(n_boot):
        # Draw one weight per cluster
        weights_map = {c: rng.choice(webb) for c in clusters}
        cluster_w   = d["bioguide_id"].map(weights_map)
        Y_boot      = base_res.fittedvalues + resid_series * cluster_w
        d_boot      = d.copy()
        d_boot["Y"] = Y_boot.values
        try:
            res_b = smf.wls(
                "Y ~ post + treat_x_post + C(bioguide_id) + C(target_word)",
                data=d_boot, weights=d_boot["match_weight"]
            ).fit()
            t_boot.append(res_b.tvalues.get("treat_x_post", np.nan))
        except Exception:
            pass

    t_boot = np.array([t for t in t_boot if not np.isnan(t)])
    p_boot = np.mean(np.abs(t_boot) >= np.abs(t_obs))
    return round(float(p_boot), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run all robustness checks against the baseline DiD spec and save results."""
    print("Loading data …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_raw  = np.load(EMB_PATH).astype(np.float32)
    meta_raw = pd.read_csv(META_PATH)

    emb_raw   = l2_norm(emb_raw)
    centroids = frame_centroids(emb_raw, meta_raw)

    # Filter helper: restrict to pre/post rows, drop Water frame, apply occurrence threshold
    def apply_filters(meta, emb, occ_min=3, drop_frames=None):
        """Apply standard sample restrictions and return outcome-augmented dataframe."""
        mask = (
            meta["aligned_period"].isin(["pre", "post"]) &
            (meta["frame"] != "Water") &
            (meta["occurrence_count"] >= occ_min)
        )
        if drop_frames:
            mask &= ~meta["frame"].isin(drop_frames)
        return add_outcome(meta[mask].reset_index(drop=True), emb[mask], centroids)

    base_df = apply_filters(meta_raw, emb_raw, occ_min=3)

    results = []

    # --- 1. Baseline ---
    res = fit_did(base_df)
    row = extract(res, "baseline_occ3")
    results.append(row)

    # --- 2. Occurrence-count thresholds ---
    for occ in [2, 5, 10]:
        df  = apply_filters(meta_raw, emb_raw, occ_min=occ)
        res = fit_did(df)
        row = extract(res, f"occ_min_{occ}")
        results.append(row)

    # --- 3. Drop Economic frame ---
    df  = apply_filters(meta_raw, emb_raw, occ_min=3, drop_frames=["Economic"])
    res = fit_did(df)
    row = extract(res, "drop_Economic")
    results.append(row)

    # --- 4. Leave-one-out: treatment speakers ---
    treat_bios = base_df[base_df["group"] == "treatment"]["bioguide_id"].unique()
    speakers   = base_df[["bioguide_id","speaker"]].drop_duplicates().set_index("bioguide_id")["speaker"]
    loo_rows = []
    for bio in sorted(treat_bios):
        df_loo = base_df[base_df["bioguide_id"] != bio]
        res    = fit_did(df_loo)
        row    = extract(res, f"loo_{bio}")
        row["dropped_bio"]     = bio
        row["dropped_speaker"] = speakers.get(bio, bio)
        loo_rows.append(row)

    loo_df = pd.DataFrame(loo_rows)
    # Annotate absolute coefficient change from baseline for influence ranking
    base_coef = results[0]["coef"]
    loo_df["delta"] = (loo_df["coef"] - base_coef).abs()
    loo_df.to_csv(OUT_DIR / "07_robustness_loo.csv", index=False)

    # --- 5. Wild cluster bootstrap ---
    p_boot = wild_cluster_bootstrap(base_df, n_boot=999)
    row = extract(fit_did(base_df), "wild_bootstrap")
    row["p_boot"] = p_boot
    row["label"]  = "wild_bootstrap"
    results.append(row)

    # --- save and summarise ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "06_robustness_main.csv", index=False)
    print(f"Saved: {OUT_DIR}/06_robustness_main.csv")
    print(f"Saved: {OUT_DIR}/07_robustness_loo.csv")


if __name__ == "__main__":
    main()
