# Author: Amir Lorvand
"""
analysis utilities for the H2 transfer experiment
compares transfered fixed angles with per-subproblem optimised angles

results are aggregated per graph and onlypositiev headroom cases are used
the comparison uses exact expected cut rahter than shot based performance
"""

import os
import sys
import csv
from collections import defaultdict

import numpy as np

RNG = np.random.default_rng(12345)
N_BOOT = 10000
OUTPUT_DIR = "analysis_results"


def fnum(x):
    """
    convert a value to float
    return None if conversion is not possible
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def boot_ci(values, n_boot=N_BOOT, alpha=0.05):
    """
    calculate a bootstrap confidence interval for the mean
    """
    v = np.asarray(values, dtype=float)

    if len(v) == 0:
        return (float("nan"), float("nan"))
    
    idx = RNG.integers(0, len(v), size=(n_boot, len(v)))
    b = np.mean(v[idx], axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    
    return (float(lo), float(hi))


def analyse(path):
    """
    analyse transferred and optimised angle quality for each graph family
    only positive headroom neighbourhoods are included
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(path) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    hp = [r for r in rows if r["headroom_positive"] == "True"]
    excluded = total - len(hp)

    families = sorted({r["family"] for r in hp})

    report = []
    report.append("PHASE C ANALYSIS  (H2 - transfer)")
    report.append("=" * 60)
    report.append(f"input: {path}")
    report.append(f"rows: {total}  (headroom-positive: {len(hp)}, "
                  f"excluded no-headroom: {excluded})")
    report.append("objective: exact expected cut (noise-free), boundary on")
    report.append("comparison: transferred WL angles vs COBYLA-optimised angles")
    report.append("")

    out_rows = []

    for fam in families:
        sub = [r for r in hp if r["family"] == fam]

        # aggregate results by graph
        gap_by_graph = defaultdict(list)
        ratio_by_graph = defaultdict(list)
        evals = []

        for r in sub:
            tr = fnum(r["transferred_expected_cut"])
            op = fnum(r["opt_expected_cut"])
            gid = r["graph_id"]

            gap_by_graph[gid].append(op - tr)

            if op > 1e-9:
                ratio_by_graph[gid].append(tr / op)
            
            evals.append(int(r["opt_evals"]))

        pg_gap = [np.mean(v) for v in gap_by_graph.values()]
        pg_ratio = [np.mean(v) for v in ratio_by_graph.values()]

        gap_mean = float(np.mean(pg_gap))
        gap_ci = boot_ci(pg_gap)
        ratio_mean = float(np.mean(pg_ratio))
        ratio_ci = boot_ci(pg_ratio)
        ev_mean = float(np.mean(evals))

        report.append(f"{fam}:  graphs={len(pg_gap)}")
        report.append(f"   transferred reaches {ratio_mean*100:.1f}% of optimised "
                      f"expected cut  (95% CI [{ratio_ci[0]*100:.1f}%, "
                      f"{ratio_ci[1]*100:.1f}%])")
        report.append(f"   quality gap (optimised - transferred) = {gap_mean:+.3f} "
                      f"(95% CI [{gap_ci[0]:+.3f}, {gap_ci[1]:+.3f}])")
        report.append(f"   optimisation cost = {ev_mean:.0f} objective evaluations "
                      f"(range {min(evals)}-{max(evals)}) vs 1 for transferred "
                      f"(~{ev_mean:.0f}x)")
        report.append("")

        out_rows.append({
            "family": fam, "graphs": len(pg_gap),
            "transferred_pct_of_optimised": round(ratio_mean * 100, 2),
            "ratio_ci_low_pct": round(ratio_ci[0] * 100, 2),
            "ratio_ci_high_pct": round(ratio_ci[1] * 100, 2),
            "quality_gap_mean": round(gap_mean, 4),
            "gap_ci_low": round(gap_ci[0], 4),
            "gap_ci_high": round(gap_ci[1], 4),
            "opt_evals_mean": round(ev_mean, 1),
            "opt_evals_min": min(evals), "opt_evals_max": max(evals),
        })

    # save outputs
    csv_path = os.path.join(OUTPUT_DIR, "analysis_h2_transfer.csv")
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    txt_path = os.path.join(OUTPUT_DIR, "analysis_h2_summary.txt")
    text = "\n".join(report)
    
    open(txt_path, "w").write(text)
    
    print(text)
    print(f"\nwrote {csv_path}, {txt_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/qaoa_opt_rows.csv"
    analyse(path)
