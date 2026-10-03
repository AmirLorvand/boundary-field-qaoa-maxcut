# Author: Amir Lorvand
"""
neighbourhood selection calibration for appendix A
compares single draw with best of N exact improvement and boundary norm
no QAOA is used, the comparison measures repair headroom and boundary structure

reproducibility:
graph seeds 0-4, initial assignment seed = graph seed + 1000
draw seed = graph seed * 100 + draw index
ER probability = 3 / (n - 1)
"""

import os
import csv

import numpy as np
import networkx as nx

from local_search import local_search
from neighbourhood import (select_neighbourhood,
                           generate_candidate_neighbourhoods,
                           partition_edges)
from boundary import boundary_fields
from exact_repair import exact_repair

# calibration settings
OUTPUT_DIR = "."
GRAPH_SEEDS = list(range(5))
NEIGH_PER_OPT = 10 
NUM_CANDIDATES = 6 
CELLS = [("3-regular", 100, 12),
         ("ER", 50, 12)] 

def make_graph(family, n, seed):
    """
    generate a random initial spin assignment
    """
    if family == "3-regular":
        return nx.random_regular_graph(3, n, seed=int(seed))
    return nx.gnp_random_graph(n, 3.0 / (n - 1), seed=int(seed))

def random_assignment(G, seed):
    """
    generate a random initial spin assignment
    """
    rng = np.random.default_rng(seed)

    return {v: int(rng.choice([-1, 1])) for v in G.nodes}

def eval_S(G, S, z):
    """
    evaluate repair headroom and boundary structure for one neighbourhood
    """
    parts = partition_edges(G, S)
    res = exact_repair(G, S, z)
    h = boundary_fields(G, S, z, boundary_on=True)

    return {
        "boundary_edges": len(parts["boundary_edges"]),
        "boundary_norm": float(np.linalg.norm(h)),
        "exact_improvement": res["exact_improvement"],
        "positive": res["exact_improvement"] > 1e-9,
    }

def summarise(rows):
    """
    summarise repairability and boundary statistics
    """
    n = len(rows)
    pos = [r for r in rows if r["positive"]]
    frac = len(pos) / n if n else 0.0
    pi = [r["exact_improvement"] for r in pos]

    return {
        "positive_rate": round(frac, 3),
        "no_headroom_rate": round(1 - frac, 3),
        "mean_imp_pos": round(float(np.mean(pi)), 2) if pi else 0.0,
        "median_imp_pos": round(float(np.median(pi)), 1) if pi else 0.0,
        "mean_boundary_edges": round(float(np.mean([r["boundary_edges"] for r in rows])), 1),
        "mean_boundary_norm": round(float(np.mean([r["boundary_norm"] for r in rows])), 2),
    }

def run_cell(family, n, k):
    """
    compare the three neighbourhood selection methods for one setting
    """
    cur, boc_imp, boc_bnd = [], [], []

    for gs in GRAPH_SEEDS:
        G = make_graph(family, n, gs)
        z0 = random_assignment(G, gs + 1000)
        z_star, _ = local_search(G, z0, seed=gs)

        for r in range(NEIGH_PER_OPT):
            sd = gs * 100 + r

            # single neighbourhood draw
            cur.append(eval_S(G, select_neighbourhood(G, z_star, k=k, seed=sd), z_star))

            # use the same candidate set for both best of N rules
            cands = generate_candidate_neighbourhoods(G, z_star, k, NUM_CANDIDATES, seed=sd)
            evals = [eval_S(G, S, z_star) for S in cands]

            boc_imp.append(max(evals, key=lambda e: (e["positive"], e["exact_improvement"])))
            boc_bnd.append(max(evals, key=lambda e: e["boundary_norm"]))

    return cur, boc_imp, boc_bnd

def main():
    """
    run the calibration and save the appendix A comparison results
    """
    out_rows = []
    
    header = f"{'cell':<18}{'method':<26}{'pos%':>7}{'noHR%':>7}{'meanImp':>9}{'medImp':>8}{'bEdge':>7}{'bNorm':>7}"
    
    print(header)
    print("-" * len(header))

    for family, n, k in CELLS:
        cur, bi, bb = run_cell(family, n, k)
        label = f"{family} n{n} k{k}"

        for name, rows in [("single draw", cur),
                           ("best-of-N (exact improvement)", bi),
                           ("best-of-N (boundary norm)", bb)]:
            
            s = summarise(rows)
            s.update({"cell": label, "method": name, "num_candidates": NUM_CANDIDATES})
            
            out_rows.append(s)
            
            print(f"{label:<18}{name:<26}{s['positive_rate']*100:>6.1f}%"
                  f"{s['no_headroom_rate']*100:>6.1f}%{s['mean_imp_pos']:>9.2f}"
                  f"{s['median_imp_pos']:>8.1f}{s['mean_boundary_edges']:>7.1f}"
                  f"{s['mean_boundary_norm']:>7.2f}")
        print()

    csv_path = os.path.join(OUTPUT_DIR, "neighbourhood_selection_check.csv")
    
    cols = ["cell", "method", "num_candidates", "positive_rate", "no_headroom_rate",
            "mean_imp_pos", "median_imp_pos", "mean_boundary_edges", "mean_boundary_norm"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        
        for row in out_rows:
            w.writerow({c: row[c] for c in cols})
    
    print(f"wrote {csv_path}")

if __name__ == "__main__":
    main()




