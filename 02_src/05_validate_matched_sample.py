#!/usr/bin/env python3
"""
Hard checks on matched_sample.csv (Phase 2 output).
Exit code 0 if all pass; nonzero if any check fails (print reasons).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def fail(msg: str) -> None:
    """Print a FAIL message to stderr."""
    print(f"FAIL: {msg}", file=sys.stderr)


def main() -> int:
    """Run all validation checks; return exit code 0 if all pass."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Path to matched_sample.csv (default: repo 03_output/phase2_matching/matched_sample.csv)",
    )
    p.add_argument(
        "--min-treated",
        type=int,
        default=int(os.environ.get("MIN_N_TREATED", "28")),
        help="Minimum distinct treated bioguide_ids (default from MIN_N_TREATED env or 28)",
    )
    p.add_argument(
        "--ideology-caliper",
        type=float,
        default=0.1,
        help="Absolute ideology difference tolerance vs paired treated",
    )
    p.add_argument(
        "--tenure-caliper",
        type=float,
        default=float(os.environ.get("TENURE_CALIPER_TOL", "2")),
        help="Absolute house_tenure_at_event difference tolerance (default 3 for phase2_matching.R main)",
    )
    p.add_argument(
        "--cohort-caliper",
        type=float,
        default=float(os.environ.get("COHORT_CALIPER_TOL", "1")),
        help="Absolute cohort difference tolerance (default 3 to match phase2_matching.R main; use 1 for paper spec)",
    )
    p.add_argument(
        "--min-pre-congresses",
        type=int,
        default=int(os.environ.get("MIN_PRE_CONGRESSES", "0")),
        help="If >0, each row must list at least this many distinct pre congresses (0=skip)",
    )
    p.add_argument(
        "--min-post-congresses",
        type=int,
        default=int(os.environ.get("MIN_POST_CONGRESSES", "0")),
        help="If >0, each row must list at least this many distinct post congresses (0=skip)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = Path(args.csv) if args.csv else root / "03_output/phase2_matching/matched_sample.csv"
    if not path.is_file():
        fail(f"file not found: {path}")
        return 2

    df = pd.read_csv(path, dtype={"bioguide_id": str, "matched_pair_id": int})
    errs = 0

    # 1) party match within pair
    for pid, g in df.groupby("matched_pair_id"):
        parties = g["party"].dropna().unique()
        if len(parties) != 1:
            fail(f"pair {pid}: party mismatch {parties.tolist()}")
            errs += 1

    # 2) caliper post-hoc vs paired treated
    treat = df[df["treated_or_control"] == "treated"].set_index("matched_pair_id")
    ctrl = df[df["treated_or_control"] == "control"]
    for _, r in ctrl.iterrows():
        pid = int(r["matched_pair_id"])
        if pid not in treat.index:
            fail(f"control {r['bioguide_id']}: no treated for pair {pid}")
            errs += 1
            continue
        t = treat.loc[pid]
        if isinstance(t, pd.DataFrame):
            t = t.iloc[0]
        for col, cal in [
            ("dw_nom_dim1_baseline", args.ideology_caliper),
            ("house_tenure_at_event", args.tenure_caliper),
            ("cohort", args.cohort_caliper),
        ]:
            dv = abs(float(r[col]) - float(t[col]))
            if dv > cal + 1e-9:
                fail(
                    f"pair {pid} control {r['bioguide_id']}: |{col}| diff {dv:.4g} > caliper {cal}"
                )
                errs += 1

    # 3) no bioguide in both roles
    tset = set(df.loc[df["treated_or_control"] == "treated", "bioguide_id"])
    cset = set(df.loc[df["treated_or_control"] == "control", "bioguide_id"])
    both = tset & cset
    if both:
        fail(f"bioguide_id appears as both treated and control: {sorted(both)[:20]}")
        errs += 1

    # 4) periods: pseudo_event strictly between pre max and post min (congress integers)
    def parse_cong(s: object) -> list[int]:
        """Parse a semicolon-separated congress string into a list of ints."""
        if pd.isna(s) or str(s).strip() in ("", "nan"):
            return []
        out: list[int] = []
        for x in str(s).split(";"):
            x = x.strip()
            if not x or x.lower() == "nan":
                continue
            out.append(int(x))
        return out

    for _, r in df.iterrows():
        pe = int(r["pseudo_event_congress"])
        pre = parse_cong(r["pre_period_congresses"])
        post = parse_cong(r["post_period_congresses"])
        if pre and max(pre) >= pe:
            fail(f"{r['bioguide_id']} pair {r['matched_pair_id']}: pre max {max(pre)} >= pseudo {pe}")
            errs += 1
        if post and min(post) < pe:
            fail(f"{r['bioguide_id']} pair {r['matched_pair_id']}: post min {min(post)} < pseudo {pe}")
            errs += 1
        if pre and post and set(pre) & set(post):
            fail(
                f"{r['bioguide_id']} pair {r['matched_pair_id']}: pre/post overlap "
                f"{sorted(set(pre) & set(post))[:10]}"
            )
            errs += 1
        if args.min_pre_congresses > 0 and len(pre) < args.min_pre_congresses:
            fail(
                f"{r['bioguide_id']} pair {r['matched_pair_id']}: "
                f"len(pre)={len(pre)} < --min-pre-congresses {args.min_pre_congresses}"
            )
            errs += 1
        if args.min_post_congresses > 0 and len(post) < args.min_post_congresses:
            fail(
                f"{r['bioguide_id']} pair {r['matched_pair_id']}: "
                f"len(post)={len(post)} < --min-post-congresses {args.min_post_congresses}"
            )
            errs += 1

    # 5) power floor
    nt = df.loc[df["treated_or_control"] == "treated", "bioguide_id"].nunique()
    if nt < args.min_treated:
        fail(f"N_treated unique={nt} < required {args.min_treated}")
        errs += 1

    if errs:
        print(f"Completed with {errs} failing sub-check(s).")
        return 1
    print("All validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
