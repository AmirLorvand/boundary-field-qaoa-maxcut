# Author: Amir Lorvand
"""
full repair loop utilities for chapter 3.10
this module cnnects the local search, boundary field, QAOA, warm start,
angle-scalin, and sampling components into one algorithm
"""
from dataclasses import dataclass

@dataclass
class RepairConfig:
    """
    configuration for one QAOA repair experiment
    """
    k: int
    p: int 
    shots: int 
    boundary_on: bool 
    angle_source: str 
    epsilon: float 
    max_rounds: int 
    seed: int 
    num_candidates: int = 6
    weight_attr: str = "weight"

import numpy as np 
from maxcut import cut_value 
from neighbourhood import select_neighbourhood, partition_edges, generate_candidate_neighbourhoods 
from boundary import build_sub_hamiltonian, boundary_fields 
from exact_repair import exact_repair 
from warm_start import warm_start_state, incumbent_bitstring, warmstart_overlap 
from angles import get_deployed_angles 
from qaoa import qaoa_state 
from sampling import sample_and_accept_repair 
from local_search import local_search 

def select_best_neighbourhood(G, z: dict, config: RepairConfig,
                              seed: int) -> dict:
    """
    best-of-candidates neighbourhood selection on exact improvement.

    Draws config.num_candidates candidate neighbourhoods, runs exact_repair on
    each, and selects the candidate with the largest positive exact improvement.
    If no candidate has positive improvement, the best-scoring candidate is kept
    and the call is flagged as having no headroom.

    Selection uses exact improvement only; it does not depend on the boundary
    switch, so the boundary-aware and no-boundary operators act on the same S.

    returns a dict with the selected neighbourhood and selection diagnostics.
    """
    candidates = generate_candidate_neighbourhoods(
        G, z, config.k, config.num_candidates, seed=seed,
    )

    candidate_improvements = []
    for S in candidates:
        result = exact_repair(G, S, z, weight_attr=config.weight_attr)
        candidate_improvements.append(result["exact_improvement"])

    # pick the candidate with the largest exact improvement (argmax);
    # this is the best positive one when any are positive, and the
    # least-bad otherwise.
    selected_index = int(max(range(len(candidate_improvements)),
                             key=lambda i: candidate_improvements[i]))

    selected_S = candidates[selected_index]
    selected_improvement = candidate_improvements[selected_index]
    headroom_positive = selected_improvement > 0

    return {
        "selected_S": selected_S,
        "selected_index": selected_index,
        "candidate_improvements": candidate_improvements,
        "num_candidates": len(candidates),
        "headroom_positive": headroom_positive,
    }


def run_single_repair_call(G, z: dict, config: RepairConfig,
                           round_id: int) -> tuple[dict, dict]:
    """
    run one complete QAOA repair call on the current incumbent assignment
    returns:
    1. updated assignment
    2. result row for this repair call
    """
    round_seed = config.seed + round_id

    incumbent_cut = cut_value(G, z, config.weight_attr)

    selection = select_best_neighbourhood(G, z, config, seed=round_seed)
    S = selection["selected_S"]

    edge_parts = partition_edges(G, S)

    h = boundary_fields(G, S, z, boundary_on=config.boundary_on, 
                        weight_attr=config.weight_attr)

    costs, assignments = build_sub_hamiltonian(G, S, z, 
                                               boundary_on=config.boundary_on, 
                                               weight_attr=config.weight_attr)

    exact_result = exact_repair(G, S, z, weight_attr=config.weight_attr)

    psi0 = warm_start_state(G, S, z, epsilon=config.epsilon)

    angle_data = get_deployed_angles(G, S, z, source=config.angle_source, 
                                     p=config.p, boundary_on=config.boundary_on, 
                                     weight_attr=config.weight_attr)

    psi_final = qaoa_state(costs, gammas=angle_data["deployed_gammas"], 
                           betas=angle_data["deployed_betas"], psi0=psi0)

    sampling_result = sample_and_accept_repair(G, S, z, psi_final,assignments,
                                                shots=config.shots, 
                                                seed=round_seed, 
                                                weight_attr=config.weight_attr)

    z_new = sampling_result["new_assignment"]

    row = {
        "round_id": round_id,
        "seed": round_seed,
        "boundary_on": config.boundary_on,
        "k": config.k,
        "p": config.p,
        "shots": config.shots,
        "angle_source": config.angle_source,
        "epsilon": config.epsilon,

        "incumbent_cut": incumbent_cut,
        "selected_vertices": S,
        "num_candidates": selection["num_candidates"],
        "candidate_exact_improvements": selection["candidate_improvements"],
        "selected_candidate_index": selection["selected_index"],
        "headroom_positive": selection["headroom_positive"],
        "num_interior_edges": len(edge_parts["inside_edges"]),
        "num_boundary_edges": len(edge_parts["boundary_edges"]),

        "boundary_fields": h,
        "boundary_norm": float(np.linalg.norm(h)),
        "boundary_max_abs": float(np.max(np.abs(h))) if len(h) > 0 else 0.0,

        "exact_best_cut": exact_result["best_cut"],
        "exact_improvement": exact_result["exact_improvement"],

        "s_eff": angle_data["s_eff"],
        "source_gammas": angle_data["source_gammas"].tolist(),
        "source_betas": angle_data["source_betas"].tolist(),
        "deployed_gammas": angle_data["deployed_gammas"].tolist(),
        "deployed_betas": angle_data["deployed_betas"].tolist(),

        "warmstart_bitstring": incumbent_bitstring(G, S, z),
        "warmstart_overlap": warmstart_overlap(psi_final, G, S, z),

        "best_sampled_cut": sampling_result["candidate_cut"],
        "best_sampled_improvement": sampling_result["improvement"],
        "num_improving_samples": sampling_result["num_improving_samples"],
        "improving_sample_probability": sampling_result["improving_sample_probability"],

        "accepted": sampling_result["accepted"],
        "accepted_improvement": sampling_result["improvement"] if sampling_result["accepted"] else 0.0,
        "new_cut": sampling_result["new_cut"],
    }

    return z_new, row

def repair_loop(G, z0: dict, config: RepairConfig) -> tuple[dict, list]:
    """
    run the full boundary-field QAOA repair loop
    first run classical local search,
    then repeatedly apply QAOA repair calls
    """
    z_current, local_history = local_search(G, z0, seed=config.seed, 
                                            weight_attr=config.weight_attr)

    rows = []

    for round_id in range(config.max_rounds):
        z_before = z_current.copy()
        cut_before = cut_value(G, z_before, config.weight_attr)

        z_repaired, row = run_single_repair_call(G, z_current, config,
                                                round_id=round_id)

        row["local_search_steps"] = len(local_history)

        rows.append(row)

        if row["accepted"] is False:
            break

        z_current, local_history = local_search(G, z_repaired,
                                                seed=config.seed + round_id + 1,
                                                weight_attr=config.weight_attr)

        cut_after = cut_value(G, z_current, config.weight_attr)

        if cut_after < cut_before:
            raise RuntimeError("repair loop decreased the cut value")

    return z_current, rows

