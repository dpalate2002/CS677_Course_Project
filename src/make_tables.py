"""Generate the RQ1 LaTeX table and in-text macros from injection results."""
import pandas as pd

res = pd.read_csv("results/injection_raw.csv")
errs = ["missing_required", "impossible_date", "out_of_bounds", "swapped_coords",
        "invalid_district", "wrong_district", "inflated_response", "near_duplicate"]
elab = ["Missing required", "Impossible date", "Out-of-bounds", "Swapped lat/lng",
        "Invalid district", "Wrong (valid) district", "Inflated response", "Near-duplicate"]
dets = ["Rules", "Spatial kNN", "Duplicate detector", "Isolation Forest", "LOF"]
dshort = ["Rules", "kNN", "Dup.", "IF", "LOF"]

g = res.groupby(["rate", "error", "detector"])[["precision", "recall", "f1"]].agg(["mean", "std"])


def cell(rate, e, d):
    p = g.loc[(rate, e, d), ("precision", "mean")]
    r = g.loc[(rate, e, d), ("recall", "mean")]
    f = g.loc[(rate, e, d), ("f1", "mean")]
    best = max(g.loc[(rate, e, x), ("f1", "mean")] for x in dets)
    s = f"{p:.2f}/{r:.2f}/{f:.2f}"
    return f"\\textbf{{{s}}}" if f == best and f > 0 else s


lines = [r"\begin{table}[tbp]", r"\centering\scriptsize",
         r"\caption{Precision / recall / F1 per detector and error type, 1\% of records injected "
         r"(mean of five seeds, all s.d.\ $\le$ 0.04). IF and LOF flag a fixed 5\% of records, so their precision "
         r"is capped at 0.20. Bold: best F1 per error type. 5\% results are in \texttt{injection\_summary.csv}.}",
         r"\label{tab:inj}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l" + "c" * len(dets) + r"@{}}", r"\toprule",
         "Error type & " + " & ".join(dshort) + r" \\", r"\midrule"]
for e, lab in zip(errs, elab):
    lines.append(lab + " & " + " & ".join(cell(0.01, e, d) for d in dets) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

m = lambda e, d, k="recall", rate=0.01: g.loc[(rate, e, d), (k, "mean")]
macros = [
    f"\\newcommand{{\\INFLIF}}{{{m('inflated_response','Isolation Forest'):.2f}}}",
    f"\\newcommand{{\\WRONGKNN}}{{{m('wrong_district','Spatial kNN'):.2f}}}",
    f"\\newcommand{{\\DUPREC}}{{{m('near_duplicate','Duplicate detector'):.2f}}}",
]
open("report/injection_table.tex", "w").write("\n".join(macros + lines) + "\n")

# max std check, printed for the text
sd = g.xs(0.05, level="rate")
print("max recall sd:", sd[("recall", "std")].max().round(3), "max f1 sd:", sd[("f1", "std")].max().round(3))
print(sd[sd[("f1", "std")] > 0.03])
for rate in [0.01, 0.05]:
    print(rate)
    print(g.xs(rate, level="rate")[[("precision", "mean"), ("recall", "mean"), ("f1", "mean")]]
          .round(2).unstack("detector").to_string())
