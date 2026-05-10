"""
Align treatment embeddings to pre/post periods.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR   = REPO_ROOT / "01_data" / "04_Embeddings"
MATCH_DIR = REPO_ROOT / "03_output" / "phase2_matching"

META_PATH    = EMB_DIR / "target_word_speaker_period_metadata.csv"
EMB_PATH     = EMB_DIR / "target_word_speaker_period_embeddings.npy"
MATCHED_PATH = MATCH_DIR / "matched_sample.csv"

OUT_META_PATH     = EMB_DIR / "aligned_metadata.csv"
OUT_EMB_PATH      = EMB_DIR / "aligned_embeddings.npy"
OUT_REPORT_PATH   = EMB_DIR / "alignment_coverage_report.csv"


def parse_congress_set(raw) -> set[str]:
    """Parse a semicolon-separated congress list into a set of strings."""
    if pd.isna(raw):
        return set()
    return {str(x).strip() for x in str(raw).split(";") if str(x).strip()}


def assign_aligned_period(row: pd.Series, pre_set: set, post_set: set, pseudo: str) -> str:
    """Map a row's congress-level period string to pre/post/exclude using per-speaker windows."""
    period = str(row["period"]).strip()
    if period == pseudo:
        return "exclude"
    if period in pre_set:
        return "pre"
    if period in post_set:
        return "post"
    # Congress found in neither window — outside the panel window
    return "exclude"


def build_treatment_lookup(matched: pd.DataFrame) -> dict:
    """Build {bioguide_id: {pre_set, post_set, pseudo_str}} for treated speakers."""
    treated = matched[matched["treated_or_control"] == "treated"].copy()
    lookup = {}
    for _, row in treated.iterrows():
        bio = row["bioguide_id"]
        lookup[bio] = {
            "pre_set":    parse_congress_set(row["pre_period_congresses"]),
            "post_set":   parse_congress_set(row["post_period_congresses"]),
            "pseudo":     str(int(row["pseudo_event_congress"])),
            "match_weight": row["match_weight"],
            "cohort":     row["cohort"],
        }
    return lookup


def align(meta: pd.DataFrame, treatment_lookup: dict) -> pd.DataFrame:
    """Add aligned_period column; handle control rows in-place."""
    aligned = []
    for _, row in meta.iterrows():
        group = row["group"]

        if group == "control":
            # Control is already labeled pre/post
            period = str(row["period"]).strip()
            if period in ("pre", "post"):
                aligned.append(period)
            else:
                aligned.append("exclude")  # shouldn't occur

        elif group == "treatment":
            bio = row["bioguide_id"]
            if bio not in treatment_lookup:
                # Unmatched treatment speaker
                aligned.append("exclude")
            else:
                info = treatment_lookup[bio]
                aligned.append(
                    assign_aligned_period(row, info["pre_set"], info["post_set"], info["pseudo"])
                )
        else:
            aligned.append("exclude")

    meta = meta.copy()
    meta["aligned_period"] = aligned
    return meta


def coverage_report(meta: pd.DataFrame) -> pd.DataFrame:
    """Per-speaker summary: how many pre/post/exclude rows each speaker has."""
    rows = []
    for (group, bio, speaker), grp in meta.groupby(["group", "bioguide_id", "speaker"]):
        counts = grp["aligned_period"].value_counts().to_dict()
        pre_n   = counts.get("pre", 0)
        post_n  = counts.get("post", 0)
        excl_n  = counts.get("exclude", 0)
        did_eligible = pre_n > 0 and post_n > 0
        rows.append({
            "group":        group,
            "bioguide_id":  bio,
            "speaker":      speaker,
            "pre_rows":     pre_n,
            "post_rows":    post_n,
            "excluded_rows": excl_n,
            "did_eligible": did_eligible,
        })
    return pd.DataFrame(rows).sort_values(["group", "did_eligible"], ascending=[True, False])


def main():
    """Load embeddings, assign aligned periods, write outputs, and report coverage."""
    print("Loading data …")
    meta    = pd.read_csv(META_PATH)
    emb     = np.load(EMB_PATH)
    matched = pd.read_csv(MATCHED_PATH)

    assert len(meta) == len(emb), (
        f"Row count mismatch: metadata={len(meta)}, embeddings={len(emb)}"
    )

    # --- build per-speaker congress window lookup ---
    treatment_lookup = build_treatment_lookup(matched)

    # Warn about treatment speakers present in embeddings but absent from matched_sample
    treatment_bios  = set(meta.loc[meta["group"] == "treatment", "bioguide_id"].unique())
    unmatched = treatment_bios - set(treatment_lookup.keys())
    if unmatched:
        speakers = meta.loc[meta["bioguide_id"].isin(unmatched), ["bioguide_id", "speaker"]].drop_duplicates()
        for _, r in speakers.iterrows():
            print(f"WARNING: treatment speaker {r['bioguide_id']} ({r['speaker']}) not in matched_sample — will be excluded")

    # --- align periods and save ---
    meta_aligned = align(meta, treatment_lookup)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    meta_aligned.to_csv(OUT_META_PATH, index=False)
    np.save(OUT_EMB_PATH, emb)      # rows are in identical order to metadata

    report = coverage_report(meta_aligned)
    report.to_csv(OUT_REPORT_PATH, index=False)

    print(f"Saved: {OUT_META_PATH}")
    print(f"Saved: {OUT_EMB_PATH}")
    print(f"Saved: {OUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
