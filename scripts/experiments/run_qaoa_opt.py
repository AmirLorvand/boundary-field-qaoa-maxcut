# Author: Amir Lorvand
"""
experiment runner for the H2 transferred angle comparison
optimises QAOA angles for each selected subproblem using COBYLA
QAOA-OPT uses the same neighbourhood, incumbent, warm start and boundary field
as the transferred angle arm and is evaluated using exact expected cost
"""

import os
import csv
import json
import time
import platform

import numpy as np
import networkx as nx
from scipy.optimize import minimize

from maxcut import cut_value
from local_search import local_search
from boundary import build_sub_hamiltonian
from warm_start import warm_start_state
from qaoa import qaoa_state, expected_cost
from angles import (load_fixed_angles, subproblem_scale_coefficients,
                    effective_scale, get_deployed_angles)
from exact_repair import exact_repair
from repair import RepairConfig, select_best_neighbourhood
from sampling import state_probabilities

# experiment settings
OUTPUT_DIR = "results"
FAMILIES = {"3-regular": {"n": 100}, "ER": {"n": 50}}
K = 12
DEPTH_P = 1
ANGLE_SOURCE = "wurtz_lykov_3reg_p1"
EPSILON = 0.1
NUM_CANDIDATES = 6
N_GRAPHS = 10            
CALLS_PER_GRAPH = 8      
BASE_SEED = 1000
N_RESTARTS = 2 
COBYLA_MAXITER = 100 
SMOKE = False 

def make_graph(family, n, seed):
    """
    generate one graph from the selected graph family 
    """
    if family == "3-regular":
        return nx.random_regular_graph(3, n, seed=int(seed))
    
    return nx.gnp_random_graph(n, 3.0 / (n - 1), seed=int(seed))

def initial_assignment(G, seed):
    """
    generate a random initial spin assignment
    """
    rng = np.random.default_rng(seed)

    return {v: int(rng.choice([-1, 1])) for v in G.nodes}


def optimise_angles(costs, psi0, p, start_gammas, start_betas, rng):
    """
    optimise QAOA angles using COBYLA and exact expected cost
    start from transferred angles followed by random restarts
    """
    total_evals = [0]

    def neg_cost(x):
        total_evals[0] += 1
        g = x[:p]; b = x[p:]
        psi = qaoa_state(costs, g, b, psi0)

        return -expected_cost(costs, psi)

    starts = [np.concatenate([np.asarray(start_gammas, float),
                              np.asarray(start_betas, float)])]
    
    for _ in range(N_RESTARTS):
        # random restart: gammas in [0, pi], betas in [0, pi/2] (standard ranges)
        rg = rng.uniform(0, np.pi, size=p)
        rb = rng.uniform(0, np.pi / 2, size=p)
        starts.append(np.concatenate([rg, rb]))

    best = None

    for x0 in starts:
        res = minimize(neg_cost, x0, method="COBYLA",
                       options={"maxiter": COBYLA_MAXITER})
        
        if best is None or res.fun < best[0]:
            best = (float(res.fun), np.asarray(res.x, float))
    
    return best[0], best[1], total_evals[0] 

def main():
    """
    run the H2 optimisation experiment and save results
    """
    n_graphs = 2 if SMOKE else N_GRAPHS
    calls = 3 if SMOKE else CALLS_PER_GRAPH

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "qaoa_opt_rows.csv")
    manifest_path = os.path.join(OUTPUT_DIR, "qaoa_opt_manifest.json")

    fields = [
        "graph_id", "family", "n", "call_idx", "k", "p",
        "exact_best_cut", "exact_improvement", "headroom_positive",
        "incumbent_cut",
        "transferred_expected_cut", "transferred_best_sampled_improvement",
        "opt_expected_cut", "opt_improvement_expected",
        "opt_gammas", "opt_betas", "opt_evals",
        "quality_gap_expected",   # opt_expected_cut - transferred_expected_cut
    ]

    t0 = time.time()
    n_rows = 0 

    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fields)
        writer.writeheader()

        for family, spec in FAMILIES.items():
            n = spec["n"] 

            for gi in range(n_graphs):
                graph_seed = BASE_SEED + gi
                G = make_graph(family, n, graph_seed)
                z0 = initial_assignment(G, graph_seed + 500)
                z_star, _ = local_search(G, z0, seed=graph_seed)
                graph_id = f"{family}_n{n}_g{gi}"
                incumbent_cut = cut_value(G, z_star)

                for call_idx in range(calls):
                    # use the same neighbourhood selection as the main experiment
                    cfg = RepairConfig(
                        k=K, p=DEPTH_P, shots=0, boundary_on=True,
                        angle_source=ANGLE_SOURCE, epsilon=EPSILON,
                        max_rounds=8, seed=graph_seed * 100 + 2048,
                        num_candidates=NUM_CANDIDATES,
                    )

                    round_seed = cfg.seed + call_idx
                    sel = select_best_neighbourhood(G, z_star, cfg, seed=round_seed)
                    S = sel["selected_S"]
                    headroom = sel["headroom_positive"]

                    costs, assignments = build_sub_hamiltonian(
                        G, S, z_star, boundary_on=True)
                    exact_res = exact_repair(G, S, z_star)
                    psi0 = warm_start_state(G, S, z_star, epsilon=EPSILON)

                    # calculate expected cut using transferred angles
                    ad = get_deployed_angles(G, S, z_star, source=ANGLE_SOURCE,
                                             p=DEPTH_P, boundary_on=True)
                    psi_tr = qaoa_state(costs, ad["deployed_gammas"],
                                        ad["deployed_betas"], psi0)
                    tr_cut = expected_cost(costs, psi_tr)

                    # optimise angles from transferred and random starting points
                    rng = np.random.default_rng(round_seed + 7)
                    neg, xbest, evals = optimise_angles(
                        costs, psi0, DEPTH_P,
                        ad["deployed_gammas"], ad["deployed_betas"], rng)
                    
                    opt_cut = -neg
                    g_opt = xbest[:DEPTH_P]
                    b_opt = xbest[DEPTH_P:]

                    writer.writerow({
                        "graph_id": graph_id, "family": family, "n": n,
                        "call_idx": call_idx, "k": K, "p": DEPTH_P,
                        "exact_best_cut": exact_res["best_cut"],
                        "exact_improvement": exact_res["exact_improvement"],
                        "headroom_positive": headroom,
                        "incumbent_cut": incumbent_cut,
                        "transferred_expected_cut": round(tr_cut, 6),
                        "transferred_best_sampled_improvement": "", 
                        "opt_expected_cut": round(opt_cut, 6),
                        "opt_improvement_expected": round(opt_cut - incumbent_cut, 6),
                        "opt_gammas": list(np.round(g_opt, 6)),
                        "opt_betas": list(np.round(b_opt, 6)),
                        "opt_evals": evals,
                        "quality_gap_expected": round(opt_cut - tr_cut, 6),
                    })

                    n_rows += 1

                print(f"  {graph_id}: {calls} calls optimised")

    # save optimisation settings and environment information 
    manifest = {
        "families": FAMILIES, "k": K, "depth_p": DEPTH_P,
        "angle_source": ANGLE_SOURCE, "epsilon": EPSILON,
        "num_candidates": NUM_CANDIDATES, "n_graphs": n_graphs,
        "calls_per_graph": calls, "base_seed": BASE_SEED,
        "optimiser": "COBYLA", "n_restarts": N_RESTARTS,
        "cobyla_maxiter": COBYLA_MAXITER,
        "objective": "exact expected_cost (noise-free), boundary on",
        "start_point": "transferred WL angles + random restarts",
        "smoke": SMOKE,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__, "networkx": nx.__version__,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
        "rows_written": n_rows,
    }

    json.dump(manifest, open(manifest_path, "w"), indent=2)
     
    print(f"\nwrote {csv_path}  ({n_rows} rows)")
    print(f"wrote {manifest_path}  in {manifest['elapsed_seconds']}s")

if __name__ == "__main__":
    main()


