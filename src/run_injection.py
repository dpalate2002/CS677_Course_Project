"""RQ1: inject errors of known type into a rule-passing sample and measure
precision / recall / F1 per detector and error type (plus average precision
for the score-based ML detectors)."""
import json, os, sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
import dq

N_SAMPLE = int(os.environ.get("N_SAMPLE", 40_000))   # QUICK: N_SAMPLE=10000 SEEDS=1
RATES = [0.01, 0.05]
SEEDS = list(range(int(os.environ.get("SEEDS", 5))))
ML_FRAC = 0.05          # IF / LOF flag the top 5% of scores (fixed contamination)
DUP_THR = json.load(open("results/dup_summary.json"))["best_threshold_train"]
RAW = None

t0 = time.time()
full = dq.load()
V = dq.run_rules(full)
knn_real = dq.district_knn_flags(full)
clean = full[~V.any(axis=1).to_numpy() & ~full.is_city_dup.to_numpy() & ~knn_real]
raw_cols = list(full.columns[:23])
print("clean pool", len(clean), f"{time.time()-t0:.0f}s")


def fmt_ts(ts):
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S.000")


def inject(s: pd.DataFrame, kind: str, idx: np.ndarray, rng) -> pd.DataFrame:
    s = s.copy()
    k = len(idx)
    req = pd.to_datetime(s.loc[idx, "date_requested"])
    if kind == "impossible_date":
        back = rng.integers(1, 31, k)
        closed = req.dt.normalize() - pd.to_timedelta(back, unit="D")
        s.loc[idx, "date_closed"] = closed.dt.strftime("%Y-%m-%d")
        s.loc[idx, "case_age_days"] = -back
    elif kind == "out_of_bounds":
        opts = rng.integers(0, 3, k)
        lat = s.loc[idx, "lat"].to_numpy().copy(); lng = s.loc[idx, "lng"].to_numpy().copy()
        lat[opts == 0], lng[opts == 0] = 0.0, 0.0                       # null island
        lng[opts == 1] = -lng[opts == 1]                                 # sign flip
        m = opts == 2                                                    # elsewhere in US
        lat[m] = rng.uniform(34, 47, m.sum()); lng[m] = rng.uniform(-122, -75, m.sum())
        s.loc[idx, "lat"], s.loc[idx, "lng"] = lat, lng
    elif kind == "swapped_coords":
        s.loc[idx, ["lat", "lng"]] = s.loc[idx, ["lng", "lat"]].to_numpy()
    elif kind == "invalid_district":
        s.loc[idx, "council_district"] = rng.choice([0, 10, 11, 99], k).astype(float)
    elif kind == "wrong_district":
        cur = s.loc[idx, "council_district"].to_numpy()
        new = (cur - 1 + rng.integers(1, 9, k)) % 9 + 1                  # different valid value
        s.loc[idx, "council_district"] = new
    elif kind == "missing_required":
        col = rng.choice(["service_name", "status"], k)
        for c in ("service_name", "status"):
            s.loc[idx[col == c], c] = np.nan
    elif kind == "inflated_response":
        # plausible-looking but wrong: request date pushed 1-4 years earlier,
        # still after GID launch and with a consistent case_age (rules pass)
        shift = rng.integers(365, 1461, k)
        new_req = req - pd.to_timedelta(shift, unit="D")
        new_req = new_req.where(new_req >= dq.GID_LAUNCH + pd.Timedelta(days=30),
                                dq.GID_LAUNCH + pd.Timedelta(days=30))
        s.loc[idx, "date_requested"] = fmt_ts(new_req)
        closed = pd.to_datetime(s.loc[idx, "date_closed"])
        s.loc[idx, "case_age_days"] = (closed - new_req.dt.normalize()).dt.days
    else:
        raise ValueError(kind)
    return s


def add_near_dups(s: pd.DataFrame, k: int, rng):
    src = s.sample(k, random_state=int(rng.integers(1e9)))
    c = src.copy()
    c["service_request_id"] = s.service_request_id.max() + 1 + np.arange(k)
    c["lat"] = c.lat + rng.normal(0, 8e-5, k)          # ~9 m jitter
    c["lng"] = c.lng + rng.normal(0, 8e-5, k)
    # re-report filed 5 min to 72 h later, while the original is still open
    orig = pd.to_datetime(c.date_requested)
    closed = pd.to_datetime(c.date_closed)
    room_h = ((closed + pd.Timedelta(days=1)) - orig).dt.total_seconds() / 3600 - 0.01
    off_h = rng.uniform(0, 1, k) * (np.minimum(72, room_h) - 1 / 12) + 1 / 12
    req = orig + pd.to_timedelta(off_h, unit="h")
    c["date_requested"] = fmt_ts(req)
    c["date_closed"] = closed.dt.strftime("%Y-%m-%d")
    c["case_age_days"] = (closed - req.dt.normalize()).dt.days
    out = pd.concat([s, c], ignore_index=True)
    truth = np.r_[np.zeros(len(s), bool), np.ones(k, bool)]
    return out, truth


KINDS = ["impossible_date", "out_of_bounds", "swapped_coords", "invalid_district",
         "missing_required", "inflated_response", "wrong_district", "near_duplicate"]
rows = []
for seed in SEEDS:
    rng0 = np.random.default_rng(seed)
    samp_idx = rng0.choice(len(clean), N_SAMPLE, replace=False)
    base = clean.iloc[samp_idx][raw_cols].reset_index(drop=True)
    ref = dq.prepare(clean.drop(clean.index[samp_idx])[raw_cols])   # disjoint reference
    for rate in RATES:
        for kind in KINDS:
            rng = np.random.default_rng(seed * 1000 + int(rate * 100) * 10 + KINDS.index(kind))
            k = int(rate * N_SAMPLE)
            if kind == "near_duplicate":
                raw, truth = add_near_dups(base, k, rng)
            else:
                idx = rng.choice(N_SAMPLE, k, replace=False)
                raw = inject(base, kind, idx, rng)
                truth = np.zeros(N_SAMPLE, bool); truth[idx] = True
            d = dq.prepare(raw).reset_index(drop=True)
            preds = {}
            preds["Rules"] = dq.rule_flags(d)
            preds["Spatial kNN"] = dq.district_knn_flags(d, ref=ref)
            X = dq.features(d, ref=ref)
            s_if, s_lof = dq.ml_scores(X, seed=seed)
            preds["Isolation Forest"] = dq.top_frac(s_if, ML_FRAC)
            preds["LOF"] = dq.top_frac(s_lof, ML_FRAC)
            pairs = dq.candidate_pairs(d)
            sc = dq.score_pairs(pairs, d)
            preds["Duplicate detector"] = dq.duplicate_flags(d, DUP_THR, pairs, sc)
            preds["Rules+kNN+Dup"] = preds["Rules"] | preds["Spatial kNN"] | preds["Duplicate detector"]
            preds["All combined"] = preds["Rules+kNN+Dup"] | preds["Isolation Forest"]
            ap = {"Isolation Forest": average_precision_score(truth, s_if),
                  "LOF": average_precision_score(truth, s_lof)}
            for det, pr in preds.items():
                p, r, f = dq.prf(pr, truth)
                rows.append(dict(seed=seed, rate=rate, error=kind, detector=det,
                                 precision=p, recall=r, f1=f, flagged=int(pr.sum()),
                                 injected=int(truth.sum()), ap=ap.get(det, np.nan)))
            print(f"seed {seed} rate {rate} {kind:18s} "
                  + " ".join(f"{k_[:6]}={dq.prf(v, truth)[1]:.2f}" for k_, v in preds.items())
                  + f"  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv("results/injection_raw.csv", index=False)

res = pd.DataFrame(rows)
res.to_csv("results/injection_raw.csv", index=False)
agg = (res.groupby(["rate", "error", "detector"])[["precision", "recall", "f1", "ap"]]
       .agg(["mean", "std"]).round(3))
agg.to_csv("results/injection_summary.csv")
print(f"done {time.time()-t0:.0f}s")
