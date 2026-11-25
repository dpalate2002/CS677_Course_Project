"""RQ3: does cleaning change median response time per council district?
Cumulative cleaning stages, bootstrap CIs, rank changes; plus a 'naive'
cleaning that falls into the date-granularity trap."""
import json, sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from sklearn.neighbors import BallTree
import dq

t0 = time.time()
df = dq.load().reset_index(drop=True)
V = dq.run_rules(df)
log = []

# district repair / imputation from spatial neighbours
ok_ref = df.lat.notna() & df.council_district.notna() & ~V.V1_coords_in_city
knn_flag = dq.district_knn_flags(df)
ref = df[ok_ref & ~knn_flag]
tree = BallTree(np.radians(ref[["lat", "lng"]].to_numpy()), metric="haversine")
need = ((df.council_district.isna() | knn_flag) & df.lat.notna() & ~V.V1_coords_in_city).to_numpy()
_, idx = tree.query(np.radians(df.loc[need, ["lat", "lng"]].to_numpy()), k=15)
vals = ref.council_district.to_numpy()[idx]
new_cd = np.array([np.bincount(v.astype(int)).argmax() for v in vals], dtype=float)

# High-confidence unlabeled duplicates: same service, <=15 m, filed within 24 h
# while the earlier request was open, and the later report filed by a resident
# channel (crew/worker tickets are often separate work items, e.g. one per tag).
pairs = dq.candidate_pairs(df)
svc = df.service_name.to_numpy()
strict = (svc[pairs.i] == svc[pairs.j]) & (pairs.dist_m <= 15) & (pairs.dt_days <= 1)
det_dup = np.zeros(len(df), bool)
det_dup[pairs.j[strict]] = True
det_dup &= df.case_origin.isin(["Mobile", "Web", "Phone", "Email"]).to_numpy()
det_dup &= ~df.is_city_dup.to_numpy()

hard = V[["C1_required_nonnull", "V1_coords_in_city", "V3_zip_format",
          "K1_close_after_request", "T3_age_nonnegative"]].any(axis=1).to_numpy()

stages = {}
d0 = df.copy()
stages["S0 Raw"] = d0
d1 = d0[~hard]
log.append(("S1 Remove hard rule violations", int(hard.sum())))
stages["S1 −Rule violations"] = d1
d2 = d1[~d1.is_city_dup]
log.append(("S2 Remove City-labeled duplicates", int(d1.is_city_dup.sum())))
stages["S2 −City duplicates"] = d2
d3 = d2[~det_dup[d2.index]]
log.append(("S3 Remove high-confidence unlabeled resident duplicates", int(det_dup[d2.index].sum())))
stages["S3 −Detected duplicates"] = d3
d4 = d3.copy()
fix = pd.Series(new_cd, index=df.index[need])
fix = fix[fix.index.isin(d4.index)]
n_imp = int(d4.loc[fix.index, "council_district"].isna().sum())
n_fix = len(fix) - n_imp
d4.loc[fix.index, "council_district"] = fix
log.append(("S4 Impute missing districts from neighbours", n_imp))
log.append(("S4 Reassign spatially inconsistent districts", n_fix))
stages["S4 District repair"] = d4
naive = df[~(df.day_closed < df.ts_requested)]          # the trap
log.append(("Naive: drop 'closed before opened' by full timestamp", int(len(df) - len(naive))))
stages["Naive (timestamp trap)"] = naive

rng = np.random.default_rng(0)


def boot_ci(x, B=400):
    x = np.asarray(x)
    meds = np.array([np.median(rng.choice(x, len(x))) for _ in range(B)])
    return np.percentile(meds, [2.5, 97.5])


rows = []
for name, d in stages.items():
    for status_scope in ["All", "Closed only"]:
        dd = d if status_scope == "All" else d[d.status == "Closed"]
        dd = dd[dd.council_district.isin(dq.DISTRICTS)]
        for cd, g in dd.groupby("council_district"):
            a = g.case_age_days.to_numpy()
            lo, hi = boot_ci(a) if name in ("S0 Raw", "S4 District repair",
                                            "Naive (timestamp trap)") else (np.nan, np.nan)
            rows.append(dict(stage=name, scope=status_scope, district=int(cd), n=len(a),
                             median=float(np.median(a)), mean=float(a.mean()),
                             p90=float(np.percentile(a, 90)), ci_lo=lo, ci_hi=hi))
        a = dd.case_age_days.to_numpy()
        rows.append(dict(stage=name, scope=status_scope, district=0, n=len(a),
                         median=float(np.median(a)), mean=float(a.mean()),
                         p90=float(np.percentile(a, 90)), ci_lo=np.nan, ci_hi=np.nan))
    print(name, f"{time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(rows)
res.to_csv("results/district_response.csv", index=False)

# rank stability: rank districts by mean and by median (ties averaged)
rank_rows = []
for scope in ["All", "Closed only"]:
    base = res[(res.stage == "S0 Raw") & (res.scope == scope) & (res.district > 0)].set_index("district")
    for name in stages:
        cur = res[(res.stage == name) & (res.scope == scope) & (res.district > 0)].set_index("district")
        for stat in ["median", "mean", "p90"]:
            tau, _ = kendalltau(base[stat], cur.loc[base.index, stat])
            moved = int((base[stat].rank(method="min") != cur.loc[base.index, stat].rank(method="min")).sum())
            rank_rows.append(dict(scope=scope, stage=name, stat=stat, kendall_tau=tau,
                                  districts_changed_rank=moved))
pd.DataFrame(rank_rows).to_csv("results/district_rank_stability.csv", index=False)
pd.DataFrame(log, columns=["step", "records"]).to_csv("results/cleaning_log.csv", index=False)
print(pd.DataFrame(log, columns=["step", "records"]).to_string(index=False))
piv = res[(res.scope == "Closed only")].pivot(index="district", columns="stage", values="mean").round(1)
print(piv.to_string())
piv = res[(res.scope == "Closed only")].pivot(index="district", columns="stage", values="median")
print(piv.to_string())
print(pd.DataFrame(rank_rows).to_string(index=False))
print(f"{time.time()-t0:.0f}s")
