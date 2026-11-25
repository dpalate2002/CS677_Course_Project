"""Core library: loading, quality rules, features, and detectors for the
San Diego Get It Done 311 data-quality project (CS 677)."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, BallTree
from sklearn.preprocessing import StandardScaler

DATA = "data/gid2025.csv"
for _d in ("results", "figs", "report"):
    os.makedirs(_d, exist_ok=True)

# City of San Diego bounding box (generous; covers San Ysidro to Rancho Bernardo)
LAT_MIN, LAT_MAX = 32.53, 33.12
LNG_MIN, LNG_MAX = -117.31, -116.90
DISTRICTS = set(range(1, 10))
VALID_STATUS = {"Closed", "Referred"}
GID_LAUNCH = pd.Timestamp("2016-05-01")   # Get It Done went live May 2016
YEAR_START, YEAR_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
REQUIRED = ["service_request_id", "date_requested", "service_name", "status", "date_closed"]


# --------------------------------------------------------------------------- load
def load(path: str = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return prepare(df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Parse types. date_closed has no time component, so we keep a
    day-granularity copy of date_requested for all comparisons."""
    df = df.copy()
    df["ts_requested"] = pd.to_datetime(df["date_requested"], errors="coerce")
    df["day_requested"] = df["ts_requested"].dt.normalize()
    df["day_closed"] = pd.to_datetime(df["date_closed"], errors="coerce")
    df["is_city_dup"] = df["service_request_parent_id"].notna()
    return df


# -------------------------------------------------------------------------- rules
# Each rule returns a boolean Series: True = record VIOLATES the rule.
def _in_box(df):
    return df["lat"].between(LAT_MIN, LAT_MAX) & df["lng"].between(LNG_MIN, LNG_MAX)


RULES = {
    # name: (dimension, description, function)
    "C1_required_nonnull": ("Completeness", "ID, request date, service, status, close date present",
                            lambda d: d[REQUIRED].isna().any(axis=1)),
    "C2_location_present": ("Completeness", "Latitude/longitude present",
                            lambda d: d["lat"].isna() | d["lng"].isna()),
    "C3_district_present": ("Completeness", "Council district present",
                            lambda d: d["council_district"].isna()),
    "C4_zip_present": ("Completeness", "Zip code present",
                       lambda d: d["zipcode"].isna()),
    "U1_id_unique": ("Uniqueness", "service_request_id unique",
                     lambda d: d["service_request_id"].duplicated(keep=False)),
    "V1_coords_in_city": ("Validity", "Coordinates inside San Diego bounding box",
                          lambda d: d["lat"].notna() & d["lng"].notna() & ~_in_box(d)),
    "V2_district_domain": ("Validity", "Council district in 1..9",
                           lambda d: d["council_district"].notna()
                           & ~d["council_district"].isin(DISTRICTS)),
    "V3_zip_format": ("Validity", "Zip is 5-digit and starts with 919/920/921",
                      lambda d: d["zipcode"].notna() & ~(
                          (d["zipcode"] >= 91900) & (d["zipcode"] <= 92199))),
    "V4_status_domain": ("Validity", "Status in {Closed, Referred}",
                         lambda d: d["status"].notna() & ~d["status"].isin(VALID_STATUS)),
    "K1_close_after_request": ("Consistency", "day(date_requested) <= date_closed",
                               lambda d: d["day_closed"] < d["day_requested"]),
    "K2_age_matches_dates": ("Consistency", "case_age_days == date_closed - day(date_requested)",
                             lambda d: d["day_closed"].notna() & d["day_requested"].notna()
                             & ((d["day_closed"] - d["day_requested"]).dt.days
                                != d["case_age_days"])),
    "T1_closed_in_2025": ("Timeliness", "date_closed within 2025",
                          lambda d: d["day_closed"].notna()
                          & ~d["day_closed"].between(YEAR_START, YEAR_END)),
    "T2_requested_after_launch": ("Timeliness", "date_requested on/after Get It Done launch (May 2016)",
                                  lambda d: d["ts_requested"].notna()
                                  & (d["ts_requested"] < GID_LAUNCH)),
    "T3_age_nonnegative": ("Timeliness", "case_age_days >= 0",
                           lambda d: d["case_age_days"] < 0),
}

# Rules used as a *detector* (hard errors only; completeness of optional
# location fields is reported but not treated as an error by itself).
DETECTOR_RULES = ["C1_required_nonnull", "U1_id_unique", "V1_coords_in_city",
                  "V2_district_domain", "V3_zip_format", "V4_status_domain",
                  "K1_close_after_request", "K2_age_matches_dates",
                  "T1_closed_in_2025", "T2_requested_after_launch", "T3_age_nonnegative"]


def run_rules(df: pd.DataFrame, names=None) -> pd.DataFrame:
    names = names or list(RULES)
    return pd.DataFrame({n: RULES[n][2](df).fillna(False).astype(bool) for n in names},
                        index=df.index)


def rule_flags(df: pd.DataFrame) -> np.ndarray:
    return run_rules(df, DETECTOR_RULES).any(axis=1).to_numpy()


def naive_timestamp_violations(df: pd.DataFrame) -> int:
    """The format trap: comparing the date-only close field to a full timestamp."""
    return int((df["day_closed"] < df["ts_requested"]).sum())


# ---------------------------------------------------------- spatial consistency
def district_knn_flags(df: pd.DataFrame, ref: pd.DataFrame | None = None,
                       k: int = 15, agree: float = 0.8) -> np.ndarray:
    """Flag a record whose recorded district disagrees with >= `agree` of its
    k nearest neighbours (by haversine) in the reference data. Offline
    substitute for a point-in-polygon check against district boundaries."""
    ref = df if ref is None else ref
    r = ref.dropna(subset=["lat", "lng", "council_district"])
    r = r[_in_box(r)]
    tree = BallTree(np.radians(r[["lat", "lng"]].to_numpy()), metric="haversine")
    rd = r["council_district"].to_numpy()
    flags = np.zeros(len(df), dtype=bool)
    ok = (df["lat"].notna() & df["lng"].notna() & df["council_district"].notna()
          & _in_box(df)).to_numpy()
    q = df.loc[ok, ["lat", "lng"]].to_numpy()
    if len(q) == 0:
        return flags
    _, idx = tree.query(np.radians(q), k=k + 1)
    idx = idx[:, 1:]  # drop nearest (self when ref contains the record)
    nb = rd[idx]
    own = df.loc[ok, "council_district"].to_numpy()
    out = np.zeros(len(q), dtype=bool)
    for i in range(len(q)):
        vals, cnt = np.unique(nb[i], return_counts=True)
        j = cnt.argmax()
        out[i] = (vals[j] != own[i]) and (cnt[j] / k >= agree)
    flags[np.where(ok)[0]] = out
    return flags


# ------------------------------------------------------------------- features
def features(df: pd.DataFrame, ref: pd.DataFrame | None = None) -> np.ndarray:
    """Numeric feature matrix for unsupervised detectors. Encodings
    (frequency, per-service median age) are fit on `ref` (default: df)."""
    ref = df if ref is None else ref
    age = df["case_age_days"].astype(float)
    slog = np.sign(age) * np.log1p(np.abs(age))
    ref_log = np.sign(ref["case_age_days"]) * np.log1p(np.abs(ref["case_age_days"]))
    svc_med = ref_log.groupby(ref["service_name"]).median()
    svc_freq = ref["service_name"].value_counts(normalize=True)
    cd_freq = ref["council_district"].value_counts(normalize=True)
    X = pd.DataFrame({
        "lat": df["lat"], "lng": df["lng"],
        "log_age": slog,
        "age_vs_service": slog - df["service_name"].map(svc_med),
        "hour": df["ts_requested"].dt.hour,
        "dow": df["ts_requested"].dt.dayofweek,
        "req_year": df["ts_requested"].dt.year + df["ts_requested"].dt.dayofyear / 366,
        "svc_freq": df["service_name"].map(svc_freq).fillna(0),
        "cd_freq": df["council_district"].map(cd_freq).fillna(0),
    })
    med = X.median(numeric_only=True)
    return X.fillna(med).fillna(0).to_numpy(dtype=float)


def ml_scores(X: np.ndarray, seed: int = 0, k: int = 20):
    """Return anomaly scores (higher = more anomalous) for IF and LOF."""
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xs)
    s_if = -iso.score_samples(Xs)
    lof = LocalOutlierFactor(n_neighbors=k, n_jobs=-1).fit(Xs)
    s_lof = -lof.negative_outlier_factor_
    return s_if, s_lof


def top_frac(scores: np.ndarray, frac: float) -> np.ndarray:
    thr = np.quantile(scores, 1 - frac)
    return scores >= thr


# --------------------------------------------------------- duplicate detection
def haversine_m(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(h))


def candidate_pairs(df: pd.DataFrame, max_m: float = 50.0, max_days: float = 180.0,
                    cell: float = 0.0005, same_service: bool = False) -> pd.DataFrame:
    """Blocking on a ~50 m grid cell (and its neighbours), optionally also on
    service_name. Keep pairs within max_m metres and max_days days, and apply the
    temporal-overlap constraint: the later request j must be filed on or before
    the earlier request i was closed (a duplicate can only point to an open
    request). Returns positional pairs (i, j), i requested first."""
    d = df.reset_index(drop=True)
    m = d["lat"].notna() & d["lng"].notna() & d["ts_requested"].notna()
    d = d[m]
    gx = np.floor(d["lat"] / cell).astype(np.int64)
    gy = np.floor(d["lng"] / cell).astype(np.int64)
    svc = d["service_name"].fillna("?").to_numpy() if same_service else np.zeros(len(d))
    base = pd.DataFrame({"pos": d.index, "svc": svc,
                         "gx": gx.to_numpy(), "gy": gy.to_numpy()})
    out = []
    # compare each cell with itself and 4 of its 8 neighbours (avoids double counting)
    for dx, dy in [(0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
        other = base.assign(gx=base.gx - dx, gy=base.gy - dy)
        mg = base.merge(other, on=["svc", "gx", "gy"], suffixes=("_a", "_b"))
        if (dx, dy) == (0, 0):
            mg = mg[mg.pos_a < mg.pos_b]
        out.append(mg[["pos_a", "pos_b"]])
    pr = pd.concat(out, ignore_index=True)
    a, b = pr.pos_a.to_numpy(), pr.pos_b.to_numpy()
    full = df.reset_index(drop=True)
    dist = haversine_m(full.lat.to_numpy()[a], full.lng.to_numpy()[a],
                       full.lat.to_numpy()[b], full.lng.to_numpy()[b])
    ta = full.ts_requested.to_numpy()[a]
    tb = full.ts_requested.to_numpy()[b]
    dt = (tb - ta) / np.timedelta64(1, "D")
    keep = (dist <= max_m) & (np.abs(dt) <= max_days)
    a, b, dist, dt = a[keep], b[keep], dist[keep], dt[keep]
    swap = dt < 0                     # orient so i is the earlier request
    i = np.where(swap, b, a)
    j = np.where(swap, a, b)
    overlap = full.day_requested.to_numpy()[j] <= full.day_closed.to_numpy()[i]
    return pd.DataFrame({"i": i[overlap], "j": j[overlap], "dist_m": dist[overlap],
                         "dt_days": np.abs(dt)[overlap]}).reset_index(drop=True)


def score_pairs(pairs: pd.DataFrame, df: pd.DataFrame, max_m=50.0, tau_days=14.0,
                use_text=True) -> np.ndarray:
    """Similarity score in [0, 1]: closeness in space and time, same service
    type, plus address and description text similarity when available."""
    from rapidfuzz.fuzz import token_set_ratio
    full = df.reset_index(drop=True)
    s_space = 1 - pairs.dist_m / max_m
    s_time = np.exp(-pairs.dt_days / tau_days)
    svc = full.service_name.to_numpy()
    s_svc = (svc[pairs.i.to_numpy()] == svc[pairs.j.to_numpy()]).astype(float)
    score = 0.30 * s_space + 0.20 * s_time + 0.30 * s_svc
    if use_text:
        addr = full.street_address.fillna("").str.lower().to_numpy()
        desc = full.public_description.fillna("").str.lower().to_numpy()
        i, j = pairs.i.to_numpy(), pairs.j.to_numpy()
        s_addr = np.array([token_set_ratio(addr[x], addr[y]) / 100 if addr[x] and addr[y] else 0.5
                           for x, y in zip(i, j)])
        s_desc = np.array([token_set_ratio(desc[x], desc[y]) / 100 if desc[x] and desc[y] else 0.5
                           for x, y in zip(i, j)])
        score = score + 0.10 * s_addr + 0.10 * s_desc
    else:
        score = score + 0.10
    return score.to_numpy()


def duplicate_flags(df: pd.DataFrame, threshold: float, pairs=None, scores=None):
    """Record-level prediction: the later record of any pair scoring >= threshold."""
    if pairs is None:
        pairs = candidate_pairs(df)
        scores = score_pairs(pairs, df)
    flags = np.zeros(len(df), dtype=bool)
    flags[pairs.j.to_numpy()[scores >= threshold]] = True
    return flags


# -------------------------------------------------------------------- metrics
def prf(pred: np.ndarray, truth: np.ndarray):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
