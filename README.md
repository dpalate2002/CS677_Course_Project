# Rules vs. Learned Detectors — San Diego "Get It Done" 311 data quality (CS 677)

Devashish Palate, Mohammadreza Ghassemzadeh

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/cs677-get-it-done-dq/blob/main/CS677_pipeline_colab.ipynb)

## Run on Colab
Open `CS677_pipeline_colab.ipynb` (badge above, after replacing `<your-username>`),
set `REPO_URL`, and choose **Runtime → Run all**. It installs `rapidfuzz`, downloads the
data from the City portal, runs every stage, shows the figures and offers a zip
of the results. Full run ≈ 20 min; `QUICK = True` ≈ 3 min.

## Run locally
```
pip install -r requirements.txt
python src/get_data.py          # downloads data/gid2025.csv (128 MB, not in git)
```
Run everything from the repo root.

## Pipeline (run in order)
| Step | Command | Output (in `results/`, `figs/`) | Time* |
|---|---|---|---|
| 1. Profile + rules | `python src/run_profile.py` | `profile_columns.csv`, `rule_violations.csv`, `profile_summary.json`, `knn_district_flags.csv` | ~1 min |
| 2. Tests | `pytest tests -v` (unit only: `pytest tests -m "not data"`) | 28 unit tests pass; 8 of 14 data assertions fail = documented defects | <1 min |
| 3. RQ2 duplicates | `python src/run_duplicates.py` | `dup_summary.json`, `dup_threshold_sweep.csv`, `dup_fp_sample.csv` | <1 min |
| 4. RQ1 injection | `python src/run_injection.py` (quick: `N_SAMPLE=10000 SEEDS=1 python src/run_injection.py`) | `injection_raw.csv`, `injection_summary.csv` | ~15 min |
| 5. RQ3 cleaning | `python src/run_cleaning.py` | `district_response.csv`, `district_rank_stability.csv`, `cleaning_log.csv` | ~1 min |
| 6. Figures/tables | `python src/make_figs.py && python src/make_tables.py` | `figs/*.pdf/png`, `report/injection_table.tex` | <1 min |

*on a 2-core machine. Step 4 needs step 3 (it reads the tuned duplicate threshold).

## Code
- `src/dq.py` — library: loading (day-granularity date handling), the 14 rules,
  spatial kNN district check, feature builder, Isolation Forest / LOF,
  duplicate blocking + scoring, metrics.
- `tests/test_rules.py` — unit tests for every rule + data assertions.

## Key decisions
- All date comparisons at **day granularity**: `date_closed` has no time, and a
  naive timestamp comparison falsely flags 111,985 rows.
- City duplicate labels = non-null `service_request_parent_id` (41,999).
- Duplicate candidates must satisfy **temporal overlap** (later report filed on/before
  the earlier one closed) — true for 99.8% of City-labeled pairs.
- Spatial district check is a kNN vote (no network access to boundary shapefiles);
  swap in point-in-polygon with official council district polygons if available.

## Report
`report/report.tex` + `report/injection_table.tex`, figures from `figs/`
(Overleaf: upload both .tex files and the `figs/` folder).

`results/` and `figs/` are committed so the numbers in the report can be checked
without rerunning; rerunning overwrites them. Results depend on the data snapshot
(downloaded Nov 2026, 378,669 rows) — the City updates the file, so a later
download may differ slightly.
