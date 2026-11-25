"""Figures for the report (static PNG/PDF for LaTeX)."""
import json, sys, os
sys.path.insert(0, "src")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
which = sys.argv[1:] or ["dup", "district", "map", "heatmap", "pr"]


def save(fig, name):
    fig.savefig(f"figs/{name}.pdf"); fig.savefig(f"figs/{name}.png"); plt.close(fig)


# ------------------------------------------------ RQ2 threshold sweep
if "dup" in which:
    d = pd.read_csv("results/dup_threshold_sweep.csv")
    best = json.load(open("results/dup_summary.json"))["best_threshold_train"]
    t = d[d.split == "test_H2"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for col, c, lab in [("precision", BLUE, "Precision"), ("recall", ORANGE, "Recall"),
                        ("f1", AQUA, "F1")]:
        ax.plot(t.threshold, t[col], color=c, lw=2, label=lab)
        ax.annotate(lab, (t.threshold.iloc[-1], t[col].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.axvline(best, color=MUTED, lw=1, ls="--")
    ax.text(best, 0.96, f" chosen on Jan–Jun\n threshold = {best:.2f}", color=MUTED, fontsize=7, va="top")
    ax.set_xlabel("Pair-score threshold"); ax.set_ylabel("Score vs. City labels (Jul–Dec)")
    ax.set_ylim(0, 1); ax.set_xlim(0.3, 1.0)
    ax.legend(loc="lower left", fontsize=7, ncol=3)
    save(fig, "dup_sweep")

# ------------------------------------------------ RQ3 district dumbbell
if "district" in which:
    r = pd.read_csv("results/district_response.csv")
    r = r[(r.scope == "All") & (r.district > 0)]
    stages = [("S0 Raw", MUTED, "o", "Raw"),
              ("S4 District repair", BLUE, "o", "Cleaned (S1–S4)"),
              ("Naive (timestamp trap)", RED, "D", "Naive cleaning (trap)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, stat, lab in [(axes[0], "median", "Median days to close"),
                          (axes[1], "mean", "Mean days to close")]:
        piv = r.pivot(index="district", columns="stage", values=stat)
        y = np.arange(len(piv))[::-1]
        ax.hlines(y, piv[[s for s, *_ in stages]].min(axis=1), piv[[s for s, *_ in stages]].max(axis=1),
                  color=GRID, lw=2, zorder=1)
        for k, (s, c, m, name) in enumerate(stages):
            off = (k - 1) * 0.18
            ax.scatter(piv[s], y + off, color=c, marker=m, s=30, zorder=3, label=name,
                       edgecolor="white", linewidth=0.8)
        ax.set_yticks(y); ax.set_yticklabels([f"D{int(i)}" for i in piv.index])
        ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.18), ncol=3, fontsize=8)
    save(fig, "district_response")

# ------------------------------------------------ map of flagged records
if "map" in which:
    import dq
    df = dq.load()
    V = dq.run_rules(df)
    knn = dq.district_knn_flags(df)
    inb = df.lat.between(dq.LAT_MIN, dq.LAT_MAX) & df.lng.between(dq.LNG_MIN, dq.LNG_MAX)
    bg = df[inb & df.council_district.notna()].sample(60000, random_state=0)
    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    ax.scatter(bg.lng, bg.lat, s=0.3, color="#cfcfcf", lw=0, rasterized=True, label="All requests (sample)")
    k = df[knn]
    ax.scatter(k.lng, k.lat, s=10, color=ORANGE, edgecolor="white", lw=0.4,
               label=f"District disagrees with neighbours ({len(k)})")
    md = df[inb & df.council_district.isna()]
    ax.scatter(md.lng, md.lat, s=6, color=BLUE, edgecolor="white", lw=0.3,
               label=f"District missing, coords valid ({len(md)})")
    ax.set_aspect(1 / np.cos(np.radians(32.8)))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7, markerscale=1.5)
    save(fig, "map_flags")

# ------------------------------------------------ RQ1 heatmap
if "heatmap" in which:
    from matplotlib.colors import LinearSegmentedColormap
    res = pd.read_csv("results/injection_raw.csv")
    errs = ["missing_required", "impossible_date", "out_of_bounds", "swapped_coords",
            "invalid_district", "wrong_district", "inflated_response", "near_duplicate"]
    elab = ["Missing required", "Impossible date", "Out-of-bounds", "Swapped lat/lng",
            "Invalid district", "Wrong (valid) district", "Inflated response", "Near-duplicate"]
    dets = ["Rules", "Spatial kNN", "Duplicate detector", "Isolation Forest", "LOF", "All combined"]
    dlab = ["Rules", "Spatial\nkNN", "Dup.\ndetector", "Isolation\nForest", "LOF", "All\ncombined"]
    cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9dc8f7", "#338ced", "#1757ac", "#0e3d7e"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, metric, title in [(axes[0], "recall", "Recall"), (axes[1], "f1", "F1")]:
        m = (res[res.rate == 0.01].groupby(["detector", "error"])[metric].mean()
             .unstack().loc[dets, errs]).T
        ax.imshow(m.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                ax.text(j, i, "1" if v >= 0.995 else f"{v:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=7, color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(dets))); ax.set_xticklabels(dlab, fontsize=6.8)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(errs))); ax.set_yticklabels(elab if ax is axes[0] else [], fontsize=7.5)
        ax.set_xlabel(title, fontsize=9, color=INK)
        ax.tick_params(length=0); ax.grid(False)
        ax.axvline(4.5, color="white", lw=3)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    fig.tight_layout()
    save(fig, "injection_heatmap")

# ------------------------------------------------ IF vs LOF average precision
if "pr" in which:
    res = pd.read_csv("results/injection_raw.csv")
    errs = ["missing_required", "impossible_date", "out_of_bounds", "swapped_coords",
            "invalid_district", "wrong_district", "inflated_response", "near_duplicate"]
    short = ["Missing", "Imposs. date", "Out of bounds", "Swapped", "Invalid district",
             "Wrong district", "Inflated resp.", "Near-dup"]
    a = res[res.detector.isin(["Isolation Forest", "LOF"])]
    g = a.groupby(["rate", "detector", "error"]).ap.agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, rate in zip(axes, [0.01, 0.05]):
        x = np.arange(len(errs))
        for k, (det, c) in enumerate([("Isolation Forest", BLUE), ("LOF", ORANGE)]):
            s = g[(g.rate == rate) & (g.detector == det)].set_index("error").loc[errs]
            ax.bar(x + (k - 0.5) * 0.38, s["mean"], 0.36, yerr=s["std"], color=c, label=det,
                   error_kw=dict(lw=0.8, ecolor=MUTED))
        ax.axhline(rate, color=MUTED, lw=1, ls=":")
        ax.text(len(errs) - 0.5, rate, "random", color=MUTED, fontsize=6.5, va="bottom", ha="right")
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.5, rotation=40, ha="right")
        ax.set_title(f"{int(rate*100)}% injection rate", fontsize=8.5)
        ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Average precision")
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "ml_avg_precision")
print("ok", which)
