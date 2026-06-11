# Congressional Speech Framing — DiD Replication

Difference-in-differences study of whether U.S. legislators change how they frame
immigration after moving from the House to the Senate. Treated = House→Senate
"switchers"; controls = matched House-only members. Outcome = cosine similarity of
immigration target-word embeddings to frame centroids.

## Reproduce the poster table and figures

Everything needed is committed (matched sample + embeddings in
`01_data/04_Embeddings/`). The raw corpus and a GPU are **not** required.

Requirements: Python 3.10+ (`numpy pandas scipy statsmodels matplotlib pyarrow`)
and R 4.x (`dagitty`). Run from the repo root, in order:

```bash
Rscript 02_src/01_dag_verification.R         # identification DAG
python3 02_src/10_did_analysis.py            # MAIN TABLE -> did_results/02_*  (+ 00_*)
python3 02_src/11_robustness_checks.py       # robustness (slow: ~7 min, 999-draw bootstrap)
python3 02_src/12_speaker_level_did.py       # speaker-level DiD + randomization inference
python3 02_src/14_heterogeneity_analysis.py  # heterogeneity -> fig05
python3 02_src/15_model_selection_final.py   # model selection -> fig10
python3 02_src/13_parallel_trends_and_viz.py # fig01–fig04
python3 02_src/16_finalize_outputs.py        # remaining CSVs + fig06–fig09
```

Outputs go to `03_output/did_results/` (tables) and `03_output/figures/` (figures).
Sanity check after step 10: `pooled coef = 0.00600, se = 0.00294, p = 0.0415`.

## Outputs → scripts

| Output | Script |
|---|---|
| Identification DAG | `01_dag_verification.R` |
| Main DiD table (`02_did_regression_results.csv`) | `10_did_analysis.py` |
| Frame coefficients (`03_speaker_level_did.csv`) | `12_speaker_level_did.py` |
| `fig01`–`fig04` (means, frame coefs, event study, parallel trends) | `13_parallel_trends_and_viz.py` |
| `fig05` heterogeneity | `14_heterogeneity_analysis.py` |
| `fig10` model comparison | `15_model_selection_final.py` |
| `fig06`–`fig09` (robustness, t-test, FDR, LOO) | `16_finalize_outputs.py` |
| Matching balance | `04_phase2_matching.R` |

## Full rebuild from raw (optional)

Needs the 1.2 GB raw corpus (git-ignored) and a GPU for step 08. Run
`02 → 03 → 04 → 05 → 06 → 07 → 08 → 09`, then the steps above.
`did_embeddings.npy` / `did_metadata.csv` are the DiD-ready subset of `aligned_*`
(`frame != "Water"`, `occurrence_count >= 3`, speakers `F000444`/`S000033`
excluded); they are committed, so the reproduction above needs no rebuild.
