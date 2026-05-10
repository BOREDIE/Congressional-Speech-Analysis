from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TREATMENT_INDEX = ROOT / "01_data/02_Panel/transition_index.csv"
CORPUS_CANDIDATES = [
    ROOT / "02_src/congress_speech_corpus_updated0430.csv",
    ROOT / "01_data/01_RawCorpus/congress_speech_corpus_updated0430.csv",
    ROOT / "01_data/01_RawCorpus/congress_speech_corpus.csv",
]
OUTPUT_FILE = ROOT / "01_data/02_Panel/treatment_speech_corpus.csv"


def transition_phase(rows: pd.DataFrame) -> pd.Series:
    """
    Label each speech by House vs Senate tenure using transition_index.

    ``pre``  — House speeches with congress <= last_H_congress.
    ``post`` — Senate speeches with congress >= first_S_congress.

    Rows that miss index congress bounds or contradict chamber get NaN.
    """
    chambers = rows["chamber"].astype(str).str.strip().str.upper()
    cong = pd.to_numeric(rows["congress"], errors="coerce")
    last_h = pd.to_numeric(rows["last_H_congress"], errors="coerce")
    first_s = pd.to_numeric(rows["first_S_congress"], errors="coerce")

    out = pd.Series(np.nan, index=rows.index, dtype=object)

    mask_h = chambers.isin({"H", "HOUSE"})
    mask_s = chambers.isin({"S", "SENATE"})

    ok_idx = last_h.notna() & first_s.notna()

    pre_m = mask_h & ok_idx & cong.notna() & (cong <= last_h)
    post_m = mask_s & ok_idx & cong.notna() & (cong >= first_s)

    out.loc[pre_m] = "pre"
    out.loc[post_m] = "post"
    return out


def pick_corpus_file() -> Path:
    """Return the first existing corpus candidate path."""
    for candidate in CORPUS_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No corpus file found in expected locations.")


def main() -> None:
    """Filter full corpus to treatment speakers, attach period labels, and write output CSV."""
    print("Loading treatment index and corpus …")
    if not TREATMENT_INDEX.exists():
        raise FileNotFoundError(f"Missing treatment index file: {TREATMENT_INDEX}")

    treatment_df = pd.read_csv(TREATMENT_INDEX, dtype={"bioguide_id": str})
    if "bioguide_id" not in treatment_df.columns:
        raise ValueError("transition_index.csv must contain a 'bioguide_id' column.")

    treatment_ids = set(treatment_df["bioguide_id"].dropna().astype(str).str.strip())
    treatment_ids = {x for x in treatment_ids if x}   # drop empty strings
    if not treatment_ids:
        raise ValueError("No valid treatment bioguide_id values found.")

    corpus_file = pick_corpus_file()
    corpus_df = pd.read_csv(corpus_file, dtype={"bioguide_id": str})
    if "bioguide_id" not in corpus_df.columns:
        raise ValueError(f"{corpus_file} must contain a 'bioguide_id' column.")
    if "speech_text" not in corpus_df.columns:
        raise ValueError(f"{corpus_file} must contain a 'speech_text' column.")

    filtered = corpus_df[corpus_df["bioguide_id"].astype(str).str.strip().isin(treatment_ids)].copy()

    idx_cols = [
        c
        for c in ("bioguide_id", "last_H_congress", "first_S_congress")
        if c in treatment_df.columns
    ]
    merged = filtered.merge(
        treatment_df[idx_cols],
        on="bioguide_id",
        how="left",
        validate="m:1",
    )
    merged["period"] = transition_phase(merged)
    merged = merged.drop(columns=[c for c in ("last_H_congress", "first_S_congress") if c in merged.columns])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
