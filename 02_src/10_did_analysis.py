"""
DiD analysis: semantic framing shift after House → Senate transition.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind

REPO    = Path(__file__).resolve().parents[1]
EMB_DIR = REPO / "01_data" / "04_Embeddings"
OUT_DIR = REPO / "03_output" / "did_results"

EMB_PATH  = EMB_DIR / "did_embeddings.npy"
META_PATH = EMB_DIR / "did_metadata.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_norm(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; rows with zero norm are left unchanged."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def frame_centroids(emb: np.ndarray, meta: pd.DataFrame) -> dict:
    """Compute per-frame centroid using ONLY control × pre rows."""
    ref_mask = (meta["group"] == "control") & (meta["aligned_period"] == "pre")
    ref_emb  = emb[ref_mask.values]
    ref_meta = meta[ref_mask].reset_index(drop=True)
    centroids = {}
    for frame, grp in ref_meta.groupby("frame"):
        idx = grp.index.values
        c   = ref_emb[idx].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[frame] = c / norm if norm > 0 else c
    return centroids


def compute_cosine_similarity(emb: np.ndarray, meta: pd.DataFrame,
                               centroids: dict) -> np.ndarray:
    """Dot product of each (already L2-normed) row with its frame centroid."""
    Y = np.zeros(len(meta))
    for i, (frame, row_emb) in enumerate(zip(meta["frame"], emb)):
        if frame in centroids:
            Y[i] = float(np.dot(row_emb, centroids[frame]))
        else:
            Y[i] = np.nan
    return Y


def run_did(df: pd.DataFrame, label: str = "pooled") -> pd.Series:
    """
    OLS DiD with speaker + word FE, clustered SE by bioguide_id.
    Returns a Series with the key DiD coefficient and diagnostics.
    """
    # Drop rows missing weight or Y
    d = df.dropna(subset=["Y", "match_weight"]).copy()
    if d["bioguide_id"].nunique() < 5 or d["target_word"].nunique() < 2:
        return None

    formula = "Y ~ post + treat_x_post + C(bioguide_id) + C(target_word)"
    try:
        model  = smf.wls(formula, data=d, weights=d["match_weight"])
        result = model.fit(cov_type="cluster", cov_kwds={"groups": d["bioguide_id"]})
    except Exception as e:
        print(f"  [{label}] regression failed: {e}")
        return None

    coef  = result.params.get("treat_x_post", np.nan)
    se    = result.bse.get("treat_x_post",    np.nan)
    pval  = result.pvalues.get("treat_x_post", np.nan)
    ci_lo = result.conf_int().loc["treat_x_post", 0] if "treat_x_post" in result.conf_int().index else np.nan
    ci_hi = result.conf_int().loc["treat_x_post", 1] if "treat_x_post" in result.conf_int().index else np.nan

    return pd.Series({
        "label":       label,
        "n_obs":       int(result.nobs),
        "n_speakers":  d["bioguide_id"].nunique(),
        "n_words":     d["target_word"].nunique(),
        "coef_DiD":    round(coef,  5),
        "se":          round(se,    5),
        "ci_lo":       round(ci_lo, 5),
        "ci_hi":       round(ci_hi, 5),
        "p_value":     round(pval,  4),
        "sig":         "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "")),
        "r2_adj":      round(result.rsquared_adj, 4),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run DiD pipeline: load embeddings, compute cosine Y, regress, save results."""
    print("Loading data …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emb  = np.load(EMB_PATH).astype(np.float32)
    meta = pd.read_csv(META_PATH)

    # --- normalise embeddings and compute frame centroids ---
    emb = l2_norm(emb)
    centroids = frame_centroids(emb, meta)   # built from control × pre rows only

    # --- compute outcome Y = cosine similarity to frame centroid ---
    meta = meta.copy()
    meta["Y"] = compute_cosine_similarity(emb, meta, centroids)

    # DiD dummy variables
    meta["post"]         = (meta["aligned_period"] == "post").astype(int)
    meta["treat"]        = (meta["group"] == "treatment").astype(int)
    meta["treat_x_post"] = meta["treat"] * meta["post"]   # interaction = DiD estimator

    # Ensure match_weight exists for all rows (control weights from corpus)
    meta["match_weight"] = pd.to_numeric(meta["match_weight"], errors="coerce").fillna(1.0)

    # --- pooled regression ---
    results = []
    pooled = run_did(meta, label="pooled_all_frames")
    if pooled is not None:
        results.append(pooled)

    # --- per-frame regressions ---
    for frame in sorted(meta["frame"].unique()):
        df_f = meta[meta["frame"] == frame]
        res  = run_did(df_f, label=frame)
        if res is not None:
            results.append(res)

    # --- save outputs ---
    results_df = pd.DataFrame(results)
    results_path = OUT_DIR / "02_did_regression_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    meta.to_csv(OUT_DIR / "00_did_analysis_data.csv", index=False)
    print(f"Saved: {OUT_DIR / '00_did_analysis_data.csv'}")

    # --- summary table (key results) ---
    print("\n=== SUMMARY TABLE ===")
    display_cols = ["label","n_obs","n_speakers","n_words","coef_DiD","se","p_value","sig"]
    print(results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
