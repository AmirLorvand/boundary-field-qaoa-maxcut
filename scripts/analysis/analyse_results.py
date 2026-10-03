# Author: Amir Lorvand
"""
analysis utilities for the H1 boundary effect and H3 captured gain experiments
results area aggregated per graph before statistical analysis
H1 compares paired QAOA-BF and QAOA-NB results on positive head room calls
H3 measures captured gain against the exact repair ceiling across shot budgets
"""

import os
import sys
import csv
import json
import math
from collections import defaultdict

import numpy as np

RNG = np.random.default_rng(12345)   # fixed seed for reproducibility
N_BOOT = 10000
OUTPUT_DIR = "analysis_results"     

def fnum(x):
    """
    convert a value to float
    return Non if conversion is not possible
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def load_rows(path):
    """
    load experiment result rows from CSV
    """
    with open(path) as f:
        return list(csv.DictReader(f))

def bootstrap_ci(values, stat=np.mean, n_boot=N_BOOT, alpha=0.05):
    """
    calculate a percentile bootstrap confidence interval
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    
    idx = RNG.integers(0, len(v), size=(n_boot, len(v)))
    boots = stat(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    
    return (float(lo), float(hi))

def wilcoxon_signed_rank(diffs):
    """
    run a two sided Wilcoxon signed rank test on paired differences
    zero differences are removed before the test
    """
    d = np.asarray([x for x in diffs if abs(x) > 1e-12], dtype=float)
    n = len(d)

    if n == 0:
        return (float("nan"), 0, float("nan"))
    
    ranks = _rankdata(np.abs(d))
    W_plus = np.sum(ranks[d > 0])
    W_minus = np.sum(ranks[d < 0])
    W = min(W_plus, W_minus)
    mean_W = n * (n + 1) / 4.0
    sd_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    
    if sd_W == 0:
        return (float(W), n, float("nan"))
    
    z = (W - mean_W + 0.5) / sd_W
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))

    return (float(W), n, float(min(max(p, 0.0), 1.0)))

def _rankdata(a):
    """
    assign ranks and average tied values
    """
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)

    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    cnts = np.zeros(len(counts))

    for r, g in zip(ranks, inv):
        sums[g] += r
        cnts[g] += 1

    return (sums / cnts)[inv]

def _norm_cdf(x):
    """
    calculate the standard normal cumulative distribution
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def per_graph_table(rows):
    """
    build paired BF and NB results for each graph and shot budget
    multiple calls are averaged and only positive headroom case are analysed
    """
    # pair BF and NB rows from the same repair call
    paired = defaultdict(dict)

    for r in rows:
        if r["arm"] not in ("QAOA-BF", "QAOA-NB"):
            continue

        call_idx = r.get("call_idx", "0")
        key = (r["family"], r["graph_id"], r["shots"], call_idx)
        paired[key][r["arm"]] = r

    # collect call level results before averaging by graph
    per_graph = defaultdict(lambda: {"bf_imp": [], "nb_imp": [],
                                     "bf_cap": [], "nb_cap": [],
                                     "bf_acc": [], "nb_acc": [],
                                     "n_calls": 0, "n_headroom": 0})
    
    for (fam, gid, shots, _), d in paired.items():
        if "QAOA-BF" not in d or "QAOA-NB" not in d:
            continue

        bf, nb = d["QAOA-BF"], d["QAOA-NB"]
        gkey = (fam, gid, shots)
        per_graph[gkey]["n_calls"] += 1

        if bf.get("headroom_positive") != "True":
            continue  
        
        per_graph[gkey]["n_headroom"] += 1
        ei = fnum(bf["exact_improvement"])
        b_best = fnum(bf["best_sampled_improvement"])
        n_best = fnum(nb["best_sampled_improvement"])
        b_acc = fnum(bf["accepted_improvement"])
        n_acc = fnum(nb["accepted_improvement"])

        per_graph[gkey]["bf_imp"].append(b_best)
        per_graph[gkey]["nb_imp"].append(n_best)
        per_graph[gkey]["bf_acc"].append(b_acc)
        per_graph[gkey]["nb_acc"].append(n_acc)

        if ei and ei > 0:
            per_graph[gkey]["bf_cap"].append(b_best / ei)
            per_graph[gkey]["nb_cap"].append(n_best / ei)

    out = []

    for (fam, gid, shots), v in per_graph.items():
        if v["n_headroom"] == 0:
            continue
        
        out.append({
            "family": fam, "graph_id": gid, "shots": int(shots),
            "n_calls": v["n_calls"], "n_headroom": v["n_headroom"],
            "bf_imp": float(np.mean(v["bf_imp"])),
            "nb_imp": float(np.mean(v["nb_imp"])),
            "bf_minus_nb": float(np.mean(v["bf_imp"]) - np.mean(v["nb_imp"])),
            "bf_cap": float(np.mean(v["bf_cap"])) if v["bf_cap"] else float("nan"),
            "nb_cap": float(np.mean(v["nb_cap"])) if v["nb_cap"] else float("nan"),
            "bf_acc": float(np.mean(v["bf_acc"])),
            "nb_acc": float(np.mean(v["nb_acc"])),
        })
    
    return out

def no_headroom_rate(rows):
    """
    calculate the no headroom rate from BF calls for each graph family
    """
    cnt = defaultdict(lambda: [0, 0])

    for r in rows:
        if r["arm"] != "QAOA-BF":
            continue
        
        cnt[r["family"]][1] += 1
        
        if r.get("headroom_positive") != "True":
            cnt[r["family"]][0] += 1
    
    return {fam: (c[0] / c[1] if c[1] else float("nan")) for fam, c in cnt.items()}

def analyse(path):
    """
    analyse H1 and H3 results and save summary files
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = load_rows(path)
    pg = per_graph_table(rows)
    nh = no_headroom_rate(rows)

    families = sorted({r["family"] for r in pg})
    shots_list = sorted({r["shots"] for r in pg})

    report = []
    report.append("PHASE C ANALYSIS  (H1 and H3)")
    report.append("=" * 60)
    report.append(f"input: {path}")
    report.append(f"per-graph records: {len(pg)} "
                  f"(family x graph x shot budget, headroom-positive)")
    report.append("")
    report.append("No-headroom rate (selected neighbourhoods, per family):")
    
    for fam, r in nh.items():
        report.append(f"  {fam}: {r*100:.1f}%")
    
    report.append("")

    # H1: collapse shot budgets so each graph contributes one value
    h1_rows = []
    report.append("H1 - BOUNDARY EFFECT (per-graph mean BF - NB, best-sampled)")
    report.append("-" * 60)

    for fam in families:
        sub = [r for r in pg if r["family"] == fam]
        by_graph = defaultdict(list)

        for r in sub:
            by_graph[r["graph_id"]].append(r["bf_minus_nb"])

        diffs = [float(np.mean(v)) for v in by_graph.values()]
        helped = sum(1 for d in diffs if d > 1e-9)
        hurt = sum(1 for d in diffs if d < -1e-9)
        tied = sum(1 for d in diffs if abs(d) <= 1e-9)
        mean_d = float(np.mean(diffs)) if diffs else float("nan")
        ci = bootstrap_ci(diffs)
        W, n_nz, p = wilcoxon_signed_rank(diffs)

        report.append(f"{fam}:  graphs={len(diffs)}  "
                      f"helped={helped} hurt={hurt} tied={tied}")
        report.append(f"   mean per-graph BF-NB = {mean_d:+.4f}  "
                      f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        report.append(f"   Wilcoxon (discordant n={n_nz}): W={W:.1f}, p={p:.4f}"
                      "  [supporting only]")
        report.append("")

        h1_rows.append({"family": fam, "graphs": len(diffs),
                        "helped": helped, "hurt": hurt, "tied": tied,
                        "mean_bf_minus_nb": round(mean_d, 4),
                        "ci_low": round(ci[0], 4), "ci_high": round(ci[1], 4),
                        "wilcoxon_W": round(W, 1), "wilcoxon_n": n_nz,
                        "wilcoxon_p": round(p, 4)})

    # H3: compare captured gain acros shot budgets
    h3_rows = []
    report.append("H3 - CAPTURED GAIN vs SHOT BUDGET (per-graph mean)")
    report.append("-" * 60)

    for fam in families:
        report.append(f"{fam}:")

        for shots in shots_list:
            sub = [r for r in pg if r["family"] == fam and r["shots"] == shots]
            bf_caps = [r["bf_cap"] for r in sub if not np.isnan(r["bf_cap"])]
            nb_caps = [r["nb_cap"] for r in sub if not np.isnan(r["nb_cap"])]

            if not bf_caps:
                continue

            bf_m = float(np.mean(bf_caps)); nb_m = float(np.mean(nb_caps))
            bf_ci = bootstrap_ci(bf_caps)

            report.append(f"   shots={shots:>5}  graphs={len(bf_caps):>2}  "
                          f"BF captured={bf_m:.3f} CI[{bf_ci[0]:.3f},{bf_ci[1]:.3f}]  "
                          f"NB captured={nb_m:.3f}")
            
            h3_rows.append({"family": fam, "shots": shots,
                            "graphs": len(bf_caps),
                            "bf_captured_mean": round(bf_m, 4),
                            "bf_ci_low": round(bf_ci[0], 4),
                            "bf_ci_high": round(bf_ci[1], 4),
                            "nb_captured_mean": round(nb_m, 4)})
        report.append("")

    # save outputs
    h1_path = os.path.join(OUTPUT_DIR, "analysis_h1_bf_vs_nb.csv")
    h3_path = os.path.join(OUTPUT_DIR, "analysis_h3_captured_gain.csv")
    summary_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")

    _write_csv(h1_path, h1_rows)
    _write_csv(h3_path, h3_rows)

    text = "\n".join(report)
    open(summary_path, "w").write(text)

    print(text)
    print(f"\nwrote {h1_path}, {h3_path}, {summary_path}")

def _write_csv(path, rows):
    if not rows:
        open(path, "w").write("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results_rows.csv"
    analyse(path)