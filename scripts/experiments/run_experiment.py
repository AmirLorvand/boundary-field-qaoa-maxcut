# Author: Amir Lorvand
"""
experiment runner for the H1 and H3 experiments
runs paired QAOA-BF and QAOA-NB repair calls on the same neighbourhood
LS only is recorded as the baseline and exact repair is stored as the ceiling
the shot budget are swept independently for each graph instnace
"""

import os
import csv
import json
import time
import platform
from dataclasses import asdict

import numpy as np
import networkx as nx

from maxcut import cut_value
from local_search import local_search
from repair import RepairConfig, run_single_repair_call

# experiment setting
OUTPUT_DIR = "results"

FAMILIES = {
    "3-regular": {"n": 100},
    "ER":        {"n": 50},
}
K = 12
DEPTH_P = 1
ANGLE_SOURCE = "wurtz_lykov_3reg_p1"
EPSILON = 0.1
NUM_CANDIDATES = 6
MAX_ROUNDS = 8
SHOT_BUDGETS = [32, 128, 512, 2048]    
N_GRAPHS = 20
CALLS_PER_GRAPH = 8 
BASE_SEED = 1000

# Set True for a fast smoke run
SMOKE = False

def make_graph(family, n, seed):
    """
    generate one graph from the selected graph family
    """
    if family == "3-regular":
        return nx.random_regular_graph(3, n, seed=int(seed))
    
    if family == "ER":
        return nx.gnp_random_graph(n, 3.0 / (n - 1), seed=int(seed))
    
    raise ValueError(family)

def initial_assignment(G, seed):
    """
    generate a random initial spin assignment
    """
    rng = np.random.default_rng(seed)

    return {v: int(rng.choice([-1, 1])) for v in G.nodes}

def make_config(boundary_on, shots, instance_seed):
    """
    create the repair config for one experimental arm
    """
    return RepairConfig(
        k=K, p=DEPTH_P, shots=shots, boundary_on=boundary_on,
        angle_source=ANGLE_SOURCE, epsilon=EPSILON,
        max_rounds=MAX_ROUNDS, seed=instance_seed,
        num_candidates=NUM_CANDIDATES,
    )
# fields saved in the flat csv output
CSV_FIELDS = [
    "graph_id", "family", "n", "arm", "shots", "call_idx", "round_id", "seed",
    "boundary_on", "k", "p", "angle_source", "epsilon",
    "incumbent_cut", "num_interior_edges", "num_boundary_edges",
    "boundary_norm", "boundary_max_abs",
    "exact_best_cut", "exact_improvement", "headroom_positive",
    "num_candidates", "selected_candidate_index",
    "s_eff", "warmstart_overlap",
    "best_sampled_cut", "best_sampled_improvement",
    "num_improving_samples", "improving_sample_probability",
    "accepted", "accepted_improvement", "new_cut",
]

def flatten_row(row, graph_id, family, n, arm):
    """
    convert one repair result into the flat csv format
    """
    out = {"graph_id": graph_id, "family": family, "n": n, "arm": arm}

    for k in CSV_FIELDS:
        if k in out:
            continue

        out[k] = row.get(k, "")
    
    return out

def run_paired_calls(G, z_star, graph_id, family, n, shots, instance_seed, call_idx):
    """
    run BF and NB on the same incumbent, neighbourhood and seed
    each call is independent and both arms remain directly paired
    """
    csv_rows, json_rows = [], []

    for arm, boundary_on in [("QAOA-BF", True), ("QAOA-NB", False)]:
        config = make_config(boundary_on, shots, instance_seed)

        # the same call index gives both arms the same neighbourgood seed
        _, row = run_single_repair_call(G, z_star, config, round_id=call_idx)
        
        flat = flatten_row(row, graph_id, family, n, arm)
        flat["call_idx"] = call_idx
        csv_rows.append(flat)

        jr = dict(row)
        jr.update({"graph_id": graph_id, "family": family, "n": n,
                   "arm": arm, "call_idx": call_idx})
        
        # convert numpy values before json serialisation
        json_rows.append(_jsonable(jr))

    return csv_rows, json_rows

def _jsonable(d):
    """
    convert numpy values into json compatible values
    """
    out = {}
    
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, dict):
            out[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    
    return out

def main():
    """
    run the full graph and shot budget experiment
    save flat results, full json rows and a reproducibility manifest
    """
    n_graphs = 2 if SMOKE else N_GRAPHS
    shot_budgets = [128] if SMOKE else SHOT_BUDGETS
    calls_per_graph = 3 if SMOKE else CALLS_PER_GRAPH

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "results_rows.csv")
    jsonl_path = os.path.join(OUTPUT_DIR, "results_rows.jsonl")
    manifest_path = os.path.join(OUTPUT_DIR, "run_manifest.json")

    t0 = time.time()
    n_rows = 0

    with open(csv_path, "w", newline="") as cf, open(jsonl_path, "w") as jf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for family, spec in FAMILIES.items():
            n = spec["n"]

            for gi in range(n_graphs):
                graph_seed = BASE_SEED + gi
                G = make_graph(family, n, graph_seed)
                z0 = initial_assignment(G, graph_seed + 500)
                z_star, _ = local_search(G, z0, seed=graph_seed)
                ls_cut = cut_value(G, z_star)
                graph_id = f"{family}_n{n}_g{gi}"

                # record one LS only baseline for each graph
                ls_row = {f: "" for f in CSV_FIELDS}
                ls_row.update({"graph_id": graph_id, "family": family, "n": n,
                               "arm": "LS-only", "incumbent_cut": ls_cut,
                               "new_cut": ls_cut})
                
                writer.writerow(ls_row)
                jf.write(json.dumps(ls_row) + "\n")
                n_rows += 1

                # run paired BF and NB calls across all shot budgets
                for shots in shot_budgets:
                    instance_seed = graph_seed * 100 + shots

                    for call_idx in range(calls_per_graph):
                        csv_rows, json_rows = run_paired_calls(
                            G, z_star, graph_id, family, n, shots,
                            instance_seed, call_idx)
                        
                        for r in csv_rows:
                            writer.writerow(r)

                        for r in json_rows:
                            jf.write(json.dumps(r) + "\n")

                        n_rows += len(csv_rows)

                print(f"  {graph_id}: LS cut={ls_cut:.0f}, "
                      f"{len(shot_budgets)} shot budgets done")
   
    # save configuration, seed policy and environment details 
    manifest = {
        "families": FAMILIES, "k": K, "depth_p": DEPTH_P,
        "angle_source": ANGLE_SOURCE, "epsilon": EPSILON,
        "num_candidates": NUM_CANDIDATES, "max_rounds": MAX_ROUNDS,
        "shot_budgets": shot_budgets, "n_graphs": n_graphs,
        "calls_per_graph": calls_per_graph,
        "base_seed": BASE_SEED, "smoke": SMOKE,
        "seed_policy": {
            "graph_seed": "BASE_SEED + graph_index",
            "initial_assignment": "graph_seed + 500",
            "local_search": "graph_seed",
            "instance_seed (per shots)": "graph_seed*100 + shots",
            "neighbourhood/round": "instance_seed + round_id",
        },
        "arms_run": ["LS-only", "QAOA-BF", "QAOA-NB"],
        "arms_pending": ["Exact (recorded in-row)", "QAOA-OPT (Phase B)"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "platform": platform.platform(),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
        "rows_written": n_rows,
    }

    json.dump(manifest, open(manifest_path, "w"), indent=2)

    print(f"\nwrote {csv_path}")
    print(f"wrote {jsonl_path}")
    print(f"wrote {manifest_path}")
    print(f"{n_rows} rows in {manifest['elapsed_seconds']}s")

if __name__ == "__main__":
    main()
