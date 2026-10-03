# Author: Amir Lorvand
"""
This file is the demo for the presentation video
"""
# ./myenv/bin/python3 demo_presentation.py
import sys
from pathlib import Path
import numpy as np
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.dont_write_bytecode = True

from maxcut import cut_value
from local_search import local_search, is_one_flip_local_optimum
from repair import RepairConfig, run_single_repair_call

# same core settings as the main H1a and H3 experiment
N = 100
K = 12
DEPTH_P = 1
ANGLE_SOURCE = "wurtz_lykov_3reg_p1"
EPSILON = 0.1
NUM_CANDIDATES = 6
MAX_ROUNDS = 8

# same seed policy as the main experiment
GRAPH_INDEX = 5
BASE_SEED = 1000
GRAPH_SEED = BASE_SEED + GRAPH_INDEX          
INITIAL_SEED = GRAPH_SEED + 500               
SHOT_BUDGET = 128
INSTANCE_SEED = GRAPH_SEED * 100 + SHOT_BUDGET  
CALL_INDEX = 4

def make_graph(seed: int):
    """
    generate the same 3-regular graph family used in the experiment
    """
    return nx.random_regular_graph(3, N, seed=int(seed))

def initial_assignment(G, seed: int):
    """
    generate the same random spin initialisation used in the experiment
    """
    rng = np.random.default_rng(seed)
    return {v: int(rng.choice([-1, 1])) for v in G.nodes}

def make_config(boundary_on: bool):
    """
    create the real repair configuration for one demonstration arm
    """
    return RepairConfig(
        k=K,
        p=DEPTH_P,
        shots=SHOT_BUDGET,
        boundary_on=boundary_on,
        angle_source=ANGLE_SOURCE,
        epsilon=EPSILON,
        max_rounds=MAX_ROUNDS,
        seed=INSTANCE_SEED,
        num_candidates=NUM_CANDIDATES,
    )

def fmt_number(value):
    """
    print integer valued floats cleanly otherwise use three decimals
    """
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}"

def captured_gain(row):
    """
    return sampled improvement as a fraction of exact available headroom
    """
    exact = float(row["exact_improvement"])
    if exact <= 0:
        return 0.0
    improvement = max(0.0, float(row["best_sampled_improvement"]))
    return improvement / exact

def main():
    print("=" * 68)
    print("SIMULATION DEMONSTRATION — BOUNDARY FIELD QAOA REPAIR")
    print("=" * 68)

    # generate one graph and run the real classical local search
    G = make_graph(GRAPH_SEED)
    z0 = initial_assignment(G, INITIAL_SEED)
    initial_cut = cut_value(G, z0)
    z_star, local_history = local_search(G, z0, seed=GRAPH_SEED)
    incumbent_cut = cut_value(G, z_star)
    stalled = is_one_flip_local_optimum(G, z_star)

    print("[1] CLASSICAL LOCAL SEARCH")
    print(f"    Graph: 3-regular, n={N}")
    print(f"    Initial cut: {fmt_number(initial_cut)}")
    print(f"    Local-search cut: {fmt_number(incumbent_cut)}")
    print(f"    1-flip local optimum: {stalled}")
    print(f"    Improving flips taken: {len(local_history)}\n")

    # run the real paired repair arms on the same incumbent and call seed
    _, bf_row = run_single_repair_call(
        G, z_star, make_config(boundary_on=True), round_id=CALL_INDEX
    )
    _, nb_row = run_single_repair_call(
        G, z_star, make_config(boundary_on=False), round_id=CALL_INDEX
    )

    # check for the paired design 
    assert bf_row["selected_vertices"] == nb_row["selected_vertices"]
    assert bf_row["candidate_exact_improvements"] == nb_row["candidate_exact_improvements"]
    assert bf_row["exact_improvement"] == nb_row["exact_improvement"]
    assert bf_row["seed"] == nb_row["seed"]

    print("[2] BEST OF SIX NEIGHBOURHOOD SELECTION")
    candidate_values = ", ".join(
        fmt_number(v) for v in bf_row["candidate_exact_improvements"]
    )
    print(f"    Candidate exact improvements: [{candidate_values}]")
    print(f"    Selected candidate: {bf_row['selected_candidate_index'] + 1} of {bf_row['num_candidates']}")
    print(f"    Active neighbourhood size: k={bf_row['k']}")
    print(f"    Selected vertices: {bf_row['selected_vertices']}")
    print(f"    Exact repair headroom: +{fmt_number(bf_row['exact_improvement'])}\n")

    print("[3] FROZEN EXTERIOR → BOUNDARY FIELD")
    h = [fmt_number(v) for v in bf_row["boundary_fields"]]
    print(f"    Boundary edges: {bf_row['num_boundary_edges']}")
    print(f"    Boundary field vector h: [{', '.join(h)}]")
    print(f"    ||h||: {float(bf_row['boundary_norm']):.3f}\n")

    print("[4] TRANSFERRED ANGLES, NO PER SUBPROBLEM OPTIMISATION")
    print(
        "    Source angles: "
        f"gamma={float(bf_row['source_gammas'][0]):.3f}, "
        f"beta={float(bf_row['source_betas'][0]):.3f}"
    )
    print(
        "    QAOA-BF: "
        f"s_eff={float(bf_row['s_eff']):.3f}, "
        f"deployed gamma={float(bf_row['deployed_gammas'][0]):.3f}"
    )
    print(
        "    QAOA-NB: "
        f"s_eff={float(nb_row['s_eff']):.3f}, "
        f"deployed gamma={float(nb_row['deployed_gammas'][0]):.3f}"
    )
    print(f"    Mixer beta remains {float(bf_row['deployed_betas'][0]):.3f}\n")

    print(f"[5] QAOA SAMPLING, {SHOT_BUDGET} SHOTS")
    print("    ------------------------------------------------------------")
    print("                         QAOA-NB          QAOA-BF")
    print("    ------------------------------------------------------------")
    print(
        "    Best improvement       "
        f"{fmt_number(nb_row['best_sampled_improvement']):>8}"
        f"          {fmt_number(bf_row['best_sampled_improvement']):>8}"
    )
    print(
        "    Captured gain          "
        f"{captured_gain(nb_row):>8.2f}"
        f"          {captured_gain(bf_row):>8.2f}"
    )
    print(
        "    Improving samples      "
        f"{int(nb_row['num_improving_samples']):>8}"
        f"          {int(bf_row['num_improving_samples']):>8}"
    )
    print(
        "    Repair accepted        "
        f"{str(nb_row['accepted']):>8}"
        f"          {str(bf_row['accepted']):>8}"
    )
    print("    ------------------------------------------------------------")
    print(f"    Exact available improvement: +{fmt_number(bf_row['exact_improvement'])}\n")

    print("[6] ACCEPTANCE ON THE FULL GRAPH")
    print(f"    Incumbent cut: {fmt_number(incumbent_cut)}")
    print(f"    QAOA-NB new cut: {fmt_number(nb_row['new_cut'])}")
    print(f"    QAOA-BF new cut: {fmt_number(bf_row['new_cut'])}")
   
if __name__ == "__main__":
    main()

