from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "01_data/01_RawCorpus/congress_speech_corpus_updated0430.csv"
OUT_DIR = ROOT / "01_data/02_Panel"

ELECTION_DAYS = 90   # days on each side of election date to exclude as "transition window"
MIN_PRE = 10
MIN_POST = 10
K_CONTROL = 3        # control speakers matched per treated speaker
SEED = 42


def load_data() -> pd.DataFrame:
    """Load raw corpus, parse dates, and restrict to valid House/Senate speeches."""
    df = pd.read_csv(INPUT_FILE, dtype={"bioguide_id": str, "chamber": str})
    df = df.dropna(subset=["bioguide_id"])
    df = df[df["bioguide_id"].str.strip() != ""].copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df["chamber"] = df["chamber"].str.strip().str.upper()
    df = df[df["chamber"].isin(["H", "S"])].copy()
    return df


def build_transition_index(df: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Identify H→S transitioners and return their congress span alongside house-only members."""
    # Set of chambers each member ever served in
    career = df.groupby("bioguide_id")["chamber"].apply(set)
    transitioned = career[career.apply(lambda s: {"H", "S"}.issubset(s))].index
    house_only = set(career[career.apply(lambda s: s == {"H"})].index)

    records = []
    for bio in transitioned:
        sub = df.loc[df["bioguide_id"] == bio, ["congress", "chamber"]].drop_duplicates()
        last_h = sub.loc[sub["chamber"] == "H", "congress"].max()
        first_s = sub.loc[sub["chamber"] == "S", "congress"].min()
        if last_h < first_s:  # enforce H before S ordering
            records.append(
                {"bioguide_id": bio, "last_H_congress": last_h, "first_S_congress": first_s}
            )
    return pd.DataFrame(records), house_only


def build_election_date_map(df: pd.DataFrame, transition_df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Estimate election date for each transitioner as the midpoint between last House and first Senate speech."""
    # Congress-level speech date spans
    span = (
        df.groupby(["bioguide_id", "congress"])["date"]
        .agg(["min", "max"])
        .rename(columns={"min": "start", "max": "end"})
        .reset_index()
    )
    out = {}
    for r in transition_df.itertuples(index=False):
        h_end = span.loc[
            (span["bioguide_id"] == r.bioguide_id) & (span["congress"] == r.last_H_congress), "end"
        ]
        s_start = span.loc[
            (span["bioguide_id"] == r.bioguide_id) & (span["congress"] == r.first_S_congress), "start"
        ]
        if h_end.empty or s_start.empty:
            # Fallback: use November of the election year derived from congress number
            out[r.bioguide_id] = pd.Timestamp(f"{1789 + 2 * r.first_S_congress}-11-01")
        else:
            h = pd.Timestamp(h_end.iloc[0])
            s = pd.Timestamp(s_start.iloc[0])
            out[r.bioguide_id] = h + (s - h) / 2   # midpoint as proxy for election date
    return out


def label_treatment(df: pd.DataFrame, transition_df: pd.DataFrame, event_dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Assign pre/post/excluded period labels to treatment (H→S) speeches."""
    rows = []
    w = pd.Timedelta(days=ELECTION_DAYS)
    for r in transition_df.itertuples(index=False):
        sub = df[df["bioguide_id"] == r.bioguide_id].copy()
        e = event_dates[r.bioguide_id]
        sub["treatment"] = 1
        sub["anchor_congress"] = r.last_H_congress
        sub["matched_to"] = pd.NA
        # Exclude speeches within the election window; otherwise label by chamber
        sub["period"] = np.where(
            (sub["date"] >= e - w) & (sub["date"] <= e + w),
            "excluded",
            np.where(sub["chamber"] == "H", "pre", np.where(sub["chamber"] == "S", "post", "excluded")),
        )
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def match_controls(df: pd.DataFrame, transition_df: pd.DataFrame, house_only: set[str]) -> pd.DataFrame:
    """Match up to K_CONTROL house-only controls per treated speaker on party, state, and congress overlap."""
    rng = np.random.default_rng(SEED)
    # Summarise each control-pool member by modal party/state and congress range
    pool = (
        df[df["bioguide_id"].isin(house_only)]
        .groupby("bioguide_id")
        .agg(
            party=("party", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
            state=("state", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
            min_c=("congress", "min"),
            max_c=("congress", "max"),
        )
        .reset_index()
    )
    treat = (
        df[df["bioguide_id"].isin(transition_df["bioguide_id"])]
        .groupby("bioguide_id")
        .agg(
            party=("party", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
            state=("state", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
        )
        .reset_index()
        .merge(transition_df, on="bioguide_id")
    )

    rec = []
    for t in treat.itertuples(index=False):
        # Prefer exact party + state + congress-overlap match; fall back to party only
        c = pool[
            (pool["party"] == t.party)
            & (pool["state"] == t.state)
            & (pool["min_c"] <= t.last_H_congress)
            & (pool["max_c"] >= t.last_H_congress)
        ]["bioguide_id"].tolist()
        if not c:
            c = pool[
                (pool["party"] == t.party)
                & (pool["min_c"] <= t.last_H_congress)
                & (pool["max_c"] >= t.last_H_congress)
            ]["bioguide_id"].tolist()
        if not c:
            continue
        for ctrl in rng.choice(c, size=min(K_CONTROL, len(c)), replace=False).tolist():
            rec.append(
                {"bioguide_id": ctrl, "matched_to": t.bioguide_id, "pseudo_anchor_congress": t.last_H_congress}
            )
    return pd.DataFrame(rec).drop_duplicates(subset=["bioguide_id"])


def label_controls(df: pd.DataFrame, control_df: pd.DataFrame, event_dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Assign pre/post/excluded period labels to control speeches using their matched pair's event date."""
    rows = []
    w = pd.Timedelta(days=ELECTION_DAYS)
    # Median event date used when the matched treatment speaker has no anchor
    fallback = pd.to_datetime(int(np.median([int(t.value) for t in event_dates.values()])))
    for r in control_df.itertuples(index=False):
        sub = df[df["bioguide_id"] == r.bioguide_id].copy()
        e = event_dates.get(r.matched_to, fallback)
        sub["treatment"] = 0
        sub["anchor_congress"] = r.pseudo_anchor_congress
        sub["matched_to"] = r.matched_to
        # For controls: pre = House speeches before event window, post = House speeches after
        sub["period"] = np.where(
            (sub["date"] >= e - w) & (sub["date"] <= e + w),
            "excluded",
            np.where(sub["chamber"] == "H", np.where(sub["date"] < e - w, "pre", "post"), "excluded"),
        )
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def finalize(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded rows and speakers with fewer than MIN_PRE or MIN_POST speeches."""
    panel = panel[panel["period"].isin(["pre", "post"])].copy()
    counts = panel.groupby(["bioguide_id", "period"]).size().unstack(fill_value=0)
    # Keep only speakers with sufficient pre and post observations
    valid = counts[(counts.get("pre", 0) >= MIN_PRE) & (counts.get("post", 0) >= MIN_POST)].index
    return panel[panel["bioguide_id"].isin(valid)].copy()


def main() -> None:
    print("Loading corpus …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- load and index data ---
    df = load_data()
    transition_df, house_only = build_transition_index(df)
    event_dates = build_election_date_map(df, transition_df)

    # --- label treatment and control speeches ---
    treat_sp = label_treatment(df, transition_df, event_dates)
    control_df = match_controls(df, transition_df, house_only)
    ctrl_sp = label_controls(df, control_df, event_dates)
    panel = finalize(pd.concat([treat_sp, ctrl_sp], ignore_index=True))

    # --- save outputs ---
    panel.to_parquet(OUT_DIR / "panel_speeches.parquet", index=False)
    transition_df.to_csv(OUT_DIR / "transition_index.csv", index=False)
    control_df.to_csv(OUT_DIR / "control_index.csv", index=False)

    summary = "\n".join(
        [
            "=== Panel Summary ===",
            f"Total speeches   : {len(panel):,}",
            f"Treatment (H->S) : {panel[panel['treatment'] == 1]['bioguide_id'].nunique()}",
            f"Control (H-only) : {panel[panel['treatment'] == 0]['bioguide_id'].nunique()}",
            "",
            str(panel.groupby(["treatment", "period"]).size().unstack(fill_value=0)),
        ]
    )
    (OUT_DIR / "panel_summary.txt").write_text(summary)
    print(f"Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
