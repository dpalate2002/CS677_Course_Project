"""Stage 1+2: profile the data and run the rule suite on the full dataset."""
import json, sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
import dq

t0 = time.time()
df = dq.load()
n = len(df)

# ---- column profile
rows = []
for c in df.columns[:23]:
    s = df[c]
    rows.append({"column": c, "dtype": str(s.dtype), "non_null": int(s.notna().sum()),
                 "missing_pct": round(100 * s.isna().mean(), 2),
                 "distinct": int(s.nunique()),
                 "min": s.min() if s.dtype != object and s.notna().any() else "",
                 "max": s.max() if s.dtype != object and s.notna().any() else ""})
prof = pd.DataFrame(rows)
prof.to_csv("results/profile_columns.csv", index=False)

# ---- rules
V = dq.run_rules(df)
rule_tab = pd.DataFrame({
    "rule": list(dq.RULES),
    "dimension": [dq.RULES[r][0] for r in dq.RULES],
    "description": [dq.RULES[r][1] for r in dq.RULES],
    "violations": [int(V[r].sum()) for r in dq.RULES],
})
rule_tab["pct"] = (100 * rule_tab.violations / n).round(3)
rule_tab.to_csv("results/rule_violations.csv", index=False)

knn = dq.district_knn_flags(df)
summary = {
    "n_records": n, "n_columns": 23,
    "city_labeled_duplicates": int(df.is_city_dup.sum()),
    "status_counts": df.status.value_counts().to_dict(),
    "naive_timestamp_close_before_request": dq.naive_timestamp_violations(df),
    "day_granularity_close_before_request": int(V["K1_close_after_request"].sum()),
    "records_failing_any_detector_rule": int(V[dq.DETECTOR_RULES].any(axis=1).sum()),
    "knn_district_inconsistent": int(knn.sum()),
    "requested_before_2025": int((df.ts_requested < "2025-01-01").sum()),
    "median_case_age_all": float(df.case_age_days.median()),
    "median_case_age_city_dups": float(df.loc[df.is_city_dup, "case_age_days"].median()),
    "median_case_age_non_dups": float(df.loc[~df.is_city_dup, "case_age_days"].median()),
}
json.dump(summary, open("results/profile_summary.json", "w"), indent=2, default=str)
df.loc[knn, ["service_request_id", "lat", "lng", "council_district", "comm_plan_name"]] \
  .to_csv("results/knn_district_flags.csv", index=False)
print(rule_tab.to_string(index=False))
print(json.dumps(summary, indent=2, default=str))
print(f"{time.time()-t0:.1f}s")
