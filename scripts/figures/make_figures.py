# Author: Amir Lorvand
"""
figure utilities for the H1 boundary effect and H3 captured gain results
creates figures 5.1, 5.2, 5.3 and 5.5 from the main experiment results
metrics are aggregated per graph and use positive headroom cases
captured gain is measured against the exact repair ceiling
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

# plotting styles for the QAOA arms 
STYLE = {
    "QAOA-BF": {"color": "#1f4e8c", "marker": "o", "ls": "-",  "label": "QAOA-BF (boundary)"},
    "QAOA-NB": {"color": "#c0392b", "marker": "s", "ls": "--", "label": "QAOA-NB (no boundary)"},
}

FAM_STYLE = {
    "3-regular": {"color": "#1f4e8c", "marker": "o"},
    "ER":        {"color": "#e08214", "marker": "^"},
}

RNG = np.random.default_rng(12345) 

def fnum(x):
    """
    convert a value to float
    return None if conversion is not possible
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def load_rows(path):
    """
    load experiment result rows from csv
    """
    with open(path) as f:
        return list(csv.DictReader(f))

def per_graph_table(rows):
    """
    build paired BF and NB results for each graph and shot budget
    only positive headroom calls are included
    """
    paired = defaultdict(dict)

    for r in rows:
        if r["arm"] not in ("QAOA-BF", "QAOA-NB"):
            continue

        ci = r.get("call_idx", "0")
        paired[(r["family"], r["graph_id"], r["shots"], ci)][r["arm"]] = r

    pg = defaultdict(lambda: {"bf_imp": [], "nb_imp": [], "bf_cap": [],
                              "nb_cap": [], "bnorm": []})
    
    for (fam, gid, shots, _), d in paired.items():
        if "QAOA-BF" not in d or "QAOA-NB" not in d:
            continue

        bf, nb = d["QAOA-BF"], d["QAOA-NB"]

        if bf.get("headroom_positive") != "True":
            continue

        ei = fnum(bf["exact_improvement"])
        b = fnum(bf["best_sampled_improvement"])
        n = fnum(nb["best_sampled_improvement"])
        key = (fam, gid, shots)

        pg[key]["bf_imp"].append(b)
        pg[key]["nb_imp"].append(n)
        pg[key]["bnorm"].append(fnum(bf["boundary_norm"]))

        if ei and ei > 0:
            pg[key]["bf_cap"].append(b / ei)
            pg[key]["nb_cap"].append(n / ei)

    out = []

    for (fam, gid, shots), v in pg.items():
        rec = {"family": fam, "graph_id": gid, "shots": int(shots),
               "bf_imp": float(np.mean(v["bf_imp"])),
               "nb_imp": float(np.mean(v["nb_imp"])),
               "bf_minus_nb": float(np.mean(v["bf_imp"]) - np.mean(v["nb_imp"])),
               "bnorm": float(np.mean(v["bnorm"]))}
        rec["bf_cap"] = float(np.mean(v["bf_cap"])) if v["bf_cap"] else np.nan
        rec["nb_cap"] = float(np.mean(v["nb_cap"])) if v["nb_cap"] else np.nan
        
        out.append(rec)
    
    return out

def boot_ci(vals, n_boot=10000, alpha=0.05):
    """
    calculate a bootstrap confidence interval for the mean
    """
    v = np.asarray(vals, float)

    if len(v) == 0:
        return (np.nan, np.nan)
    
    idx = RNG.integers(0, len(v), size=(n_boot, len(v)))
    b = np.mean(v[idx], axis=1)

    return tuple(np.percentile(b, [100*alpha/2, 100*(1-alpha/2)]))

def save(fig, name):
    """
    save a figure as png and pdf
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight")
        
    plt.close(fig)

def fig_5_1(pg):
    """
    plot the per graph boundary advantage
    each graph contributes one value averaged across shot budgets
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)

    for ax, fam in zip(axes, FAMILIES):
        # combine shot budgets into one boundary-effect value per graph
        by_graph = defaultdict(list)

        for r in pg:
            if r["family"] == fam:
                by_graph[r["graph_id"]].append(r["bf_minus_nb"])

        diffs = [np.mean(v) for v in by_graph.values()]

        ax.axvline(0, color="0.4", lw=1, ls=":")
        ax.hist(diffs, bins=12, color=FAM_STYLE[fam]["color"], alpha=0.8,
                edgecolor="white")
        
        m = np.mean(diffs)
        ax.axvline(m, color="black", lw=1.5, ls="-")
        ax.annotate(f"mean = {m:+.3f}  (n = {len(diffs)})",
                    xy=(m, ax.get_ylim()[1]*0.9),
                    xytext=(6, 0), textcoords="offset points", fontsize=9)
        
        ax.set_title(f"{fam}", fontsize=11)
        ax.set_xlabel("per-graph (QAOA-BF − QAOA-NB) improvement")

    axes[0].set_ylabel("number of graph instances")

    fig.suptitle("Figure 5.1 — Paired boundary effect, one value per graph "
                 "(positive = boundary helps)", fontsize=11, y=1.02)
    
    save(fig, "fig_5_1_paired_bf_nb")

def fig_5_2(pg):
    """
    plot captured gain distributions for BF and NB
    values are pooled across shot budgets for each graph family
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)

    for ax, fam in zip(axes, FAMILIES):
        bf = [r["bf_cap"] for r in pg if r["family"] == fam and not np.isnan(r["bf_cap"])]
        nb = [r["nb_cap"] for r in pg if r["family"] == fam and not np.isnan(r["nb_cap"])]
        
        bins = np.linspace(0, 1.05, 18)

        ax.hist(nb, bins=bins, color=STYLE["QAOA-NB"]["color"], alpha=0.55,
                label="QAOA-NB", edgecolor="white")
        ax.hist(bf, bins=bins, color=STYLE["QAOA-BF"]["color"], alpha=0.55,
                label="QAOA-BF", edgecolor="white")
        
        ax.axvline(np.mean(bf), color=STYLE["QAOA-BF"]["color"], lw=1.5)
        ax.axvline(np.mean(nb), color=STYLE["QAOA-NB"]["color"], lw=1.5, ls="--")
        
        ax.set_title(fam, fontsize=11)
        ax.set_xlabel("captured gain (fraction of exact ceiling)")
        ax.legend(fontsize=8, loc="upper center")

    axes[0].set_ylabel("number of per-graph records")

    fig.suptitle("Figure 5.2 — Captured-gain distribution vs the exact ceiling",
                 fontsize=11, y=1.02)
    
    save(fig, "fig_5_2_captured_gain_dist")

def fig_5_3(pg):
    """
    plot captured gain against shot budget for BF and NB
    bootstrap confidence intervals are shown around each curve
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    budgets = sorted({r["shots"] for r in pg})

    for ax, fam in zip(axes, FAMILIES):
        for arm, capkey in [("QAOA-BF", "bf_cap"), ("QAOA-NB", "nb_cap")]:
            means, los, his = [], [], []

            for s in budgets:
                vals = [r[capkey] for r in pg
                        if r["family"] == fam and r["shots"] == s
                        and not np.isnan(r[capkey])]
                
                means.append(np.mean(vals))
                lo, hi = boot_ci(vals)
                los.append(lo); his.append(hi)

            st = STYLE[arm]

            ax.plot(budgets, means, color=st["color"], marker=st["marker"],
                    ls=st["ls"], label=st["label"], lw=1.8, ms=6)
            
            ax.fill_between(budgets, los, his, color=st["color"], alpha=0.15)
        
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets); ax.set_xticklabels(budgets)
        ax.set_title(fam, fontsize=11)
        ax.set_xlabel("shot budget (log scale)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("captured gain (fraction of exact ceiling)")
    
    fig.suptitle("Figure 5.3 — Captured gain increases with shot budget; "
                 "boundary leads at every budget", fontsize=11, y=1.02)
    
    save(fig, "fig_5_3_captured_gain_vs_shots")

def fig_5_5(pg):
    """
    plot boundary field strength against the BF advantage
    includes the overall linear trend and Pearson correlation
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for fam in FAMILIES:
        xs = [r["bnorm"] for r in pg if r["family"] == fam]
        ys = [r["bf_minus_nb"] for r in pg if r["family"] == fam]

        ax.scatter(xs, ys, s=24, alpha=0.6, color=FAM_STYLE[fam]["color"],
                   marker=FAM_STYLE[fam]["marker"], label=fam, edgecolor="none")
        
    ax.axhline(0, color="0.4", lw=1, ls=":")
    
    # calculate the overall trend across both graph families
    allx = np.array([r["bnorm"] for r in pg])
    ally = np.array([r["bf_minus_nb"] for r in pg])

    if len(allx) > 2:
        b, a = np.polyfit(allx, ally, 1)
        xs = np.linspace(allx.min(), allx.max(), 50)

        ax.plot(xs, a + b*xs, color="black", lw=1.3,
                label=f"trend (slope {b:+.3f})")
        
        r = np.corrcoef(allx, ally)[0, 1]

        ax.annotate(f"Pearson r = {r:+.2f}", xy=(0.97, 0.03),
                    xycoords="axes fraction", ha="right", fontsize=9)
        
    ax.set_xlabel("boundary-field strength (per-graph mean ‖h‖)")
    ax.set_ylabel("BF advantage (QAOA-BF − QAOA-NB)")
    ax.set_title("Figure 5.5 — Does a stronger boundary field predict a larger advantage?",
                 fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, "fig_5_5_boundary_strength")

def main(path):
    """
    create and save all H1 and H3 figures
    """
    rows = load_rows(path)
    pg = per_graph_table(rows)

    fig_5_1(pg)
    fig_5_2(pg)
    fig_5_3(pg)
    fig_5_5(pg)

    print(f"wrote 4 figures (PNG + PDF) to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_rows.csv")


