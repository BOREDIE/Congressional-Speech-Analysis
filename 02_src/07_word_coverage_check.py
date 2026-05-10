"""
Word coverage checker for DiD target word re-selection.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

WORD_DETAIL   = REPO / "01_data/03_TargetWords/word_selection_detail.csv"
TREATMENT_CSV = REPO / "01_data/02_Panel/treatment_speech_corpus.csv"
CONTROL_CSV   = REPO / "01_data/02_Panel/control_corpus.csv"
MATCHED_CSV   = REPO / "03_output/phase2_matching/matched_sample.csv"
DEFAULT_OUT   = REPO / "01_data/03_TargetWords/word_coverage_by_cell.csv"

# Broad immigration keywords — a speech must contain at least one of these
# to be considered immigration-relevant.
IMMIGRATION_KEYWORDS = re.compile(
    r"\b("
    r"immigr\w*|migr\w*|alien|aliens|border|illegal|undocumented|unauthorized"
    r"|refugee|asylum|visa|citizenship|naturali\w*|amnesty|deport\w*"
    r"|smuggl\w*|traffick\w*|foreigner|foreign.born|noncitizen"
    r"|detention|detain|apprehend|repatri\w*|resettl\w*"
    r")\b",
    re.IGNORECASE,
)


def is_immigration_speech(text: str) -> bool:
    """Return True if the speech contains at least one immigration keyword."""
    if not isinstance(text, str):
        return False
    return bool(IMMIGRATION_KEYWORDS.search(text))


def word_stats(speeches: pd.DataFrame, word: str) -> tuple[set, int]:
    """
    Return (set of bioguide_ids, total occurrence count) for a word.
    Occurrences = sum of all token matches across all speeches in the cell.
    """
    if speeches.empty:
        return set(), 0
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    texts = speeches["speech_text"].fillna("")
    mask = texts.str.contains(pattern, regex=True)
    matched = texts[mask]
    speakers = set(speeches.loc[mask, "bioguide_id"].unique())
    occurrences = int(matched.apply(lambda t: len(pattern.findall(t))).sum()) if not matched.empty else 0
    return speakers, occurrences


def parse_congress_set(raw) -> set:
    """Parse a semicolon-separated congress string into a set of ints."""
    if pd.isna(raw):
        return set()
    return {int(x) for x in str(raw).split(";") if x.strip()}


def build_treatment_cells(treatment_full: pd.DataFrame, treated: pd.DataFrame) -> dict:
    """
    Split treatment corpus into pre/post using the exact congress windows
    from matched_sample, then keep only immigration-relevant speeches.
    """
    pre_speeches, post_speeches = [], []

    for _, row in treated.iterrows():
        bio      = row["bioguide_id"]
        pre_set  = parse_congress_set(row["pre_period_congresses"])
        post_set = parse_congress_set(row["post_period_congresses"])
        sp       = treatment_full[treatment_full["bioguide_id"] == bio]

        pre_speeches.append(sp[sp["congress"].isin(pre_set)])
        post_speeches.append(sp[sp["congress"].isin(post_set)])

    pre_df  = pd.concat(pre_speeches,  ignore_index=True) if pre_speeches  else pd.DataFrame()
    post_df = pd.concat(post_speeches, ignore_index=True) if post_speeches else pd.DataFrame()

    # Filter to immigration-relevant speeches only
    pre_df  = pre_df[pre_df["speech_text"].apply(is_immigration_speech)]
    post_df = post_df[post_df["speech_text"].apply(is_immigration_speech)]

    return {
        ("treatment", "pre"):  pre_df,
        ("treatment", "post"): post_df,
    }


def build_cell_index(treatment_full: pd.DataFrame, control: pd.DataFrame,
                     treated: pd.DataFrame) -> dict:
    """Combine treatment and control cells into a single {(group, period): DataFrame} dict."""
    cells = build_treatment_cells(treatment_full, treated)

    # Control corpus already has correct period labels; filter to immigration speeches
    ctrl_pre  = control[control["period"] == "pre"]
    ctrl_post = control[control["period"] == "post"]
    ctrl_pre  = ctrl_pre[ctrl_pre["speech_text"].apply(is_immigration_speech)]
    ctrl_post = ctrl_post[ctrl_post["speech_text"].apply(is_immigration_speech)]

    cells[("control", "pre")]  = ctrl_pre
    cells[("control", "post")] = ctrl_post
    return cells

def main():
    """Compute per-cell speaker coverage for each candidate word and write results CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-speakers", type=int, default=1,
        help="Minimum distinct speakers required in EACH of the 4 cells (default: 1)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    print("Loading corpora …")
    treatment_full = pd.read_csv(TREATMENT_CSV, dtype={"bioguide_id": str})
    control        = pd.read_csv(CONTROL_CSV,   dtype={"bioguide_id": str})
    matched        = pd.read_csv(MATCHED_CSV,   dtype={"bioguide_id": str})
    words_df       = pd.read_csv(WORD_DETAIL)

    # Only the 25 treatment speakers with a defined pre window
    treated = matched[
        (matched["treated_or_control"] == "treated") &
        matched["pre_period_congresses"].notna()
    ].copy()

    cells = build_cell_index(treatment_full, control, treated)

    cell_keys = [
        ("treatment", "pre"),
        ("treatment", "post"),
        ("control",   "pre"),
        ("control",   "post"),
    ]

    records = []
    total = len(words_df)
    for i, (_, row) in enumerate(words_df.iterrows(), 1):
        word  = str(row["word"]).strip()
        frame = row["frame"]

        spk, occ = {}, {}
        for key in cell_keys:
            spk[key], occ[key] = word_stats(cells[key], word)

        t_pre_spk  = len(spk[("treatment", "pre")])
        t_post_spk = len(spk[("treatment", "post")])
        c_pre_spk  = len(spk[("control",   "pre")])
        c_post_spk = len(spk[("control",   "post")])
        min_cell   = min(t_pre_spk, t_post_spk, c_pre_spk, c_post_spk)

        # Period-level totals (treatment + control combined)
        pre_occ_total  = occ[("treatment", "pre")]  + occ[("control", "pre")]
        post_occ_total = occ[("treatment", "post")] + occ[("control", "post")]
        pre_spk_total  = len(spk[("treatment", "pre")]  | spk[("control", "pre")])
        post_spk_total = len(spk[("treatment", "post")] | spk[("control", "post")])

        records.append({
            "frame":          frame,
            "word":           word,
            "original_count": row["count"],
            "in_range":       row["in_range"],
            "t_pre_spk":      t_pre_spk,
            "t_post_spk":     t_post_spk,
            "c_pre_spk":      c_pre_spk,
            "c_post_spk":     c_post_spk,
            "min_spk":        min_cell,
            "t_pre_occ":      occ[("treatment", "pre")],
            "t_post_occ":     occ[("treatment", "post")],
            "c_pre_occ":      occ[("control",   "pre")],
            "c_post_occ":     occ[("control",   "post")],
            "pre_occ_total":  pre_occ_total,
            "post_occ_total": post_occ_total,
            "pre_spk_total":  pre_spk_total,
            "post_spk_total": post_spk_total,
            "all_covered":    min_cell >= args.min_speakers,
        })

    result = pd.DataFrame(records).sort_values(
        ["all_covered", "min_spk", "frame", "word"],
        ascending=[False, False, True, True],
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
