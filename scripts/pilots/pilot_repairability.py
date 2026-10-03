# Author: Amir Lorvand
"""
unconditioned repairability pilot for appendix A
measures positive exact repair headroom using single draw neighbourhoods
used to compare graph sizes and k values before the main experiments
no QAOA is run, only local search and exact subgraph repair are used

reproducibility:
graph seeds 0-7, initial assignment seed = graph seed + 1000
neighbourhood seed = graph seed * 100 + neighbourhood index
ER probability = 3 / (n - 1)
"""

import os
import csv
import json
from collections import defaultdict

import numpy as np
import networkx as nx

from maxcut import cut_value
from local_search import local_search
from neighbourhood import select_neighbourhood, partition_edges
from exact_repair import exact_repair

# pilot setting
OUTPUT_DIR = "." 
GRAPH_SEEDS = list(range(8))
NEIGH_PER_OPT = 20
FAMILIES = ["3-regular", "ER"] 
SIZES = [50, 100] 
KS = [10, 12] 

def make_graph(family, n, seed):
    """
    generate one graph from the selected graph family
    """
    if family == "3-regular":
        return nx.random_regular_graph(3, n, seed=int(seed))
    
    if family == "ER":
        return nx.gnp_random_graph(n, 3.0 / (n - 1), seed=int(seed))
    
    raise ValueError(family)

def random_assignment(G, seed):
    """
    generate a random initial spin assignment
    """
    rng = np.random.default_rng(seed)

    return {v: int(rng.choice([-1, 1])) for v in G.nodes}

def run_cell(family, n, k, graph_seeds, neigh_per_opt):
    """
    run the repairability pilot for one graph and neighbourhood setting
    """
    rows = []

    for gs in graph_seeds:
        G = make_graph(family, n, gs)
        
        if G.number_of_nodes() < k:
            continue

        z0 = random_assignment(G, gs + 1000)
        z_star, _ = local_search(G, z0, seed=gs)

        for r in range(neigh_per_opt):
            S = select_neighbourhood(G, z_star, k=k, seed=gs * 100 + r)
            parts = partition_edges(G, S)
            res = exact_repair(G, S, z_star)

            rows.append({
                "family": family, "n": n, "k": k,
                "graph_seed": gs, "neigh": r,
                "interior_edges": len(parts["inside_edges"]),
                "boundary_edges": len(parts["boundary_edges"]),
                "exact_improvement": res["exact_improvement"],
                "positive": res["exact_improvement"] > 1e-9,
            })

    return rows

def summarise(rows):
    """
    summarise repairability and boundary statistics
    """
    n = len(rows)
    pos = [r for r in rows if r["positive"]]
    frac = len(pos) / n if n else 0.0
    pos_imp = [r["exact_improvement"] for r in pos]

    return {
        "calls": n,
        "positive_rate": round(frac, 3),
        "no_headroom_rate": round(1 - frac, 3),
        "mean_imp_when_pos": round(float(np.mean(pos_imp)), 3) if pos_imp else 0.0,
        "median_imp_when_pos": round(float(np.median(pos_imp)), 1) if pos_imp else 0.0,
        "mean_boundary_edges": round(float(np.mean([r["boundary_edges"] for r in rows])), 1) if rows else 0.0,
    }

def main():
    """
    run the pilot and save the appendix A repairability results
    """
    all_rows = []
    table = []

    header = f"{'family':<10} {'n':>4} {'k':>3} {'calls':>6} {'pos%':>7} {'meanImp':>8} {'medImp':>7} {'meanBedges':>11}"
    
    print(header)
    print("-" * len(header))

    for family in FAMILIES:
        for n in SIZES:
            for k in KS:
                rows = run_cell(family, n, k, GRAPH_SEEDS, NEIGH_PER_OPT)
                
                all_rows += rows
                s = summarise(rows)
                s.update({"family": family, "n": n, "k": k})
                table.append(s)

                print(f"{family:<10} {n:>4} {k:>3} {s['calls']:>6} "
                      f"{s['positive_rate']*100:>6.1f}% {s['mean_imp_when_pos']:>8.2f} "
                      f"{s['median_imp_when_pos']:>7.1f} {s['mean_boundary_edges']:>11.1f}")

    csv_path = os.path.join(OUTPUT_DIR, "pilot_repairability_table.csv")

    cols = ["family", "n", "k", "calls", "positive_rate", "no_headroom_rate",
            "mean_imp_when_pos", "median_imp_when_pos", "mean_boundary_edges"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for row in table:
            w.writerow({c: row[c] for c in cols})

    json_path = os.path.join(OUTPUT_DIR, "pilot_repairability_rows.json")
    json.dump(all_rows, open(json_path, "w"))

    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}  ({len(all_rows)} repair-call records)")

if __name__ == "__main__":
    main()



