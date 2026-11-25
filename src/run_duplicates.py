"""RQ2: recover City-labeled duplicates. Threshold is tuned on requests closed
Jan-Jun 2025 and evaluated on Jul-Dec 2025 (temporal hold-out)."""
import json, sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
import dq

t0 = time.time()
df = dq.load().reset_index(drop=True)
pairs = dq.candidate_pairs(df)
print("candidate pairs", len(pairs), f"{time.time()-t0:.0f}s")
scores = dq.score_pairs(pairs, df)
print("scored", f"{time.time()-t0:.0f}s")

truth = df.is_city_dup.to_numpy()
id2pos = pd.Series(df.index, index=df.service_request_id)
par = df.service_request_parent_id
has_par = par.notna() & par.isin(id2pos.index)
true_pairs = set(zip(id2pos[par[has_par].astype(int)].to_numpy(), df.index[has_par]))
pair_keys = list(zip(pairs.i, pairs.j))
pair_keys_rev = list(zip(pairs.j, pairs.i))
is_true_pair = np.array([(a in true_pairs) or (b in true_pairs) for a, b in zip(pair_keys, pair_keys_rev)])
blocking_recall = is_true_pair.sum() / len(true_pairs)

h2 = (df.day_closed >= "2025-07-01").to_numpy()
grid = np.round(np.arange(0.30, 0.96, 0.01), 2)
rows = []
for thr in grid:
    pred = dq.duplicate_flags(df, thr, pairs, scores)
    for name, m in [("train_H1", ~h2), ("test_H2", h2), ("all", np.ones_like(h2))]:
        p, r, f = dq.prf(pred[m], truth[m])
        rows.append(dict(threshold=thr, split=name, precision=p, recall=r, f1=f,
                         flagged=int(pred[m].sum())))
    sel = scores >= thr
    pp = is_true_pair[sel].sum() / max(sel.sum(), 1)
    pr_ = is_true_pair[sel].sum() / len(true_pairs)
    rows.append(dict(threshold=thr, split="pairs_all", precision=pp, recall=pr_,
                     f1=2 * pp * pr_ / (pp + pr_) if pp + pr_ else 0, flagged=int(sel.sum())))
res = pd.DataFrame(rows)
res.to_csv("results/dup_threshold_sweep.csv", index=False)
tr = res[res.split == "train_H1"]
best = float(tr.loc[tr.f1.idxmax(), "threshold"])
out = {
    "candidate_pairs": int(len(pairs)),
    "labeled_pairs_with_parent_in_data": len(true_pairs),
    "blocking_recall_pairs": float(blocking_recall),
    "best_threshold_train": best,
    "test_H2": res[(res.split == "test_H2") & (res.threshold == best)].iloc[0].to_dict(),
    "train_H1": tr[tr.threshold == best].iloc[0].to_dict(),
    "pairs_all": res[(res.split == "pairs_all") & (res.threshold == best)].iloc[0].to_dict(),
}
# unlabeled "false positives": sample for manual inspection
pred = dq.duplicate_flags(df, best, pairs, scores)
sel = (scores >= best)
fp_pairs = pairs[sel & ~is_true_pair].assign(score=scores[sel & ~is_true_pair])
fp_pairs = fp_pairs[~truth[fp_pairs.j.to_numpy()]]
samp = fp_pairs.sample(min(40, len(fp_pairs)), random_state=0)
cols = ["service_request_id", "date_requested", "service_name", "street_address",
        "public_description", "status", "case_age_days"]
insp = pd.concat([df.loc[samp.i, cols].reset_index(drop=True).add_suffix("_a"),
                  df.loc[samp.j, cols].reset_index(drop=True).add_suffix("_b"),
                  samp[["dist_m", "dt_days", "score"]].reset_index(drop=True)], axis=1)
insp.to_csv("results/dup_fp_sample.csv", index=False)
json.dump(out, open("results/dup_summary.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
print(f"{time.time()-t0:.0f}s")
