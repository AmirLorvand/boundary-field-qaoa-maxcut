# Author: Amir Lorvand
"""
figure utilities for the H2 transferred angle comparison
plots transferred quality against optimised quality and optimisation cost
results are aggregated per graph using positive-headroom cases only
"""

import os
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "figures"
FAMILIES = ["3-regular", "ER"]
FAM_STYLE = {
    "3-regular": {"color": "#1f4e8c", "marker": "o"},
    "ER":        {"color": "#e08214", "marker": "^"},
}

def fnum(x):
    """
    convert a value to float
    return None if conversion is not possible
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def per_graph(rows):
    """
    aggregate transferred quality ratio and optimisation cost per graph
    only positive headroom cases are included
    """
    hp = [r for r in rows if r["headroom_positive"] == "True"]
    ratio = defaultdict(lambda: defaultdict(list)) 
    evals = defaultdict(lambda: defaultdict(list)) 

    for r in hp:
        fam, gid = r["family"], r["graph_id"]
        tr = fnum(r["transferred_expected_cut"])
        op = fnum(r["opt_expected_cut"])

        if op and op > 1e-9:
            ratio[fam][gid].append(tr / op)
        
        evals[fam][gid].append(int(r["opt_evals"]))

    out = {}

    for fam in FAMILIES:
        out[fam] = {
            "ratio": [np.mean(v) for v in ratio[fam].values()],
            "evals": [np.mean(v) for v in evals[fam].values()],
        }

    return out

def main(path):
    """
    create and save the H2 quality cost figure
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))

    pg = per_graph(rows)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4))

    # transferred quality as a percentage of optimised quality 
    for i, fam in enumerate(FAMILIES):
        vals = np.array(pg[fam]["ratio"]) * 100
        x = np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))

        axL.scatter(x, vals, s=30, alpha=0.6, color=FAM_STYLE[fam]["color"],
                    marker=FAM_STYLE[fam]["marker"], edgecolor="none")
        
        m = vals.mean()
        axL.plot([i - 0.2, i + 0.2], [m, m], color="black", lw=2)
        axL.annotate(f"{m:.1f}%", xy=(i, m), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=10,
                     fontweight="bold")
   
    axL.axhline(100, color="0.5", ls=":", lw=1)
    axL.set_xticks(range(len(FAMILIES)))
    axL.set_xticklabels(FAMILIES)
    axL.set_ylim(85, 102)
    axL.set_ylabel("transferred quality\n(% of optimised expected cut)")
    axL.set_title("Quality retained by transfer", fontsize=11)
    axL.grid(True, axis="y", alpha=0.3)

    # optimisation cost compared with one transferred evaluation
    for i, fam in enumerate(FAMILIES):
        vals = np.array(pg[fam]["evals"])
        x = np.full(len(vals), i) + np.random.default_rng(1).uniform(-0.08, 0.08, len(vals))
        
        axR.scatter(x, vals, s=30, alpha=0.6, color=FAM_STYLE[fam]["color"],
                    marker=FAM_STYLE[fam]["marker"], edgecolor="none")
        
        m = vals.mean()
        axR.plot([i - 0.2, i + 0.2], [m, m], color="black", lw=2)
        axR.annotate(f"{m:.0f}", xy=(i, m), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=10,
                     fontweight="bold")
        
    axR.axhline(1, color="#c0392b", ls="--", lw=1.5, label="transferred (1 evaluation)")
    axR.set_yscale("log")
    axR.set_xticks(range(len(FAMILIES)))
    axR.set_xticklabels(FAMILIES)
    axR.set_ylabel("optimisation cost\n(objective evaluations, log scale)")
    axR.set_title("Cost of optimising afresh", fontsize=11)
    axR.legend(fontsize=8, loc="center right")
    axR.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 5.4 — Transfer keeps ~95% of optimised quality at a "
                 "fraction of the cost", fontsize=11, y=1.02)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_5_4_quality_cost.{ext}"),
                    dpi=200, bbox_inches="tight")
        
    plt.close(fig)
    print(f"wrote fig_5_4_quality_cost.(png|pdf) to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/qaoa_opt_rows.csv")
