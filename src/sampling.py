# Author: Amir Lorvand
"""
sampling and acceptance utilities for chapter 3.9
these functions convert a final QAOA statevector into sampled candidate moves
"""
import numpy as np 
from exact_repair import insert_subassignment
from maxcut import cut_value, validate_spin_assignment

def state_probabilities(state: np.ndarray) -> np.ndarray:
    """
    convert a statevector into a probability distribution
    """
    state = np.asarray(state, dtype=complex)

    if len(state) == 0:
        raise ValueError("state must not be empty")
    
    probabilities = np.abs(state) ** 2
    total_probability = float(np.sum(probabilities))

    if total_probability <= 0:
        raise ValueError("state has zero probability mass")
    
    return probabilities / total_probability

def sample_state_indices(state: np.ndarray, shots: int, 
                         seed: int | None = None) -> np.ndarray:
    """
    sample statevector indices according to their probabilities
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    
    probabilities = state_probabilities(state)
    rng = np.random.default_rng(seed)

    indices = rng.choice(len(probabilities), size=shots, 
                         replace=True, p=probabilities)

    return indices

def decode_sampled_indices(indices, assignments: list) -> list:
    """
    decode sampled statevector indices into spin assignment over S
    assignment must come from build_sub_hamiltonian
    """
    decoded_assignments = []

    for index in indices:
        index = int(index)

        if index < 0 or index >= len(assignments):
            raise ValueError("sampled index is outside assignment list")
        
        decoded_assignments.append(assignments[index].copy())

    return decoded_assignments

def evaluate_sampled_assignments(G, S: list, z: dict, sampled_assignments: list,
                                weight_attr: str = "weight") -> list:
    """
    insert each sampled subassignment into the full graph assignment
    and evaluate its full maxcut value
    """
    validate_spin_assignment(G, z)
    incumbent_cut = cut_value(G, z, weight_attr)
    evaluations = []

    for z_S in sampled_assignments:
        z_candidate = insert_subassignment(z, z_S)
        candidate_cut = cut_value(G, z_candidate, weight_attr)

        evaluations.append({
            "z_S": z_S.copy(),
            "full_assignment": z_candidate,
            "cut": candidate_cut,
            "improvement": candidate_cut - incumbent_cut
            })
    
    return evaluations

def best_sampled_repair(evaluations: list) -> dict:
    """
    return the sampled repair with the largest full graph cut value
    """
    if len(evaluations) == 0:
        raise ValueError("evaluations must not be empty")
    
    best = evaluations[0]

    for item in evaluations[1:]:
        if item["cut"] > best["cut"]:
            best = item

    return best

def accept_repair_if_improves(G, z: dict, best_repair: dict,
                               weight_attr: str="weight") -> dict:
    """
    accept the sampled repair only if it improves the full graph cut
    if it does not improve, keep the incumbent assignment unchanged
    """
    validate_spin_assignment(G, z)
    incumbent_cut = cut_value(G, z, weight_attr)
    candidate_cut = best_repair["cut"]
    improvement = candidate_cut - incumbent_cut

    if improvement > 0:
        accepted = True
        new_assignment = best_repair["full_assignment"].copy()
        new_cut = candidate_cut
    else:
        accepted = False
        new_assignment = z.copy()
        new_cut = incumbent_cut

    return {
        "accepted": accepted,
        "new_assignment": new_assignment,
        "incumbent_cut": incumbent_cut,
        "candidate_cut": candidate_cut,
        "new_cut": new_cut,
        "improvement": improvement,
        "best_z_S": best_repair["z_S"].copy()
    }

def sample_and_accept_repair(G, S: list, z: dict, state: np.ndarray, 
                             assignments: list, shots: int, 
                             seed: int | None = None,
                             weight_attr: str = "weight") -> dict:
    """
    sample candidate repairs from a QAOA statevector
    decode them using the assignments list
    evaluate tem on the full graph
    accept the best one only if it improves the cut
    """
    indices = sample_state_indices(state, shots=shots, seed=seed)
    sampled_assignments = decode_sampled_indices(indices, assignments)
    evaluations = evaluate_sampled_assignments(G, S, z, sampled_assignments,
                                               weight_attr=weight_attr)
    best_repair = best_sampled_repair(evaluations)
    num_improving_samples = sum(
        1 for item in evaluations
        if item["improvement"] > 0
    )

    improving_sample_probability = num_improving_samples / shots

    acceptance = accept_repair_if_improves(G, z, best_repair,
                                           weight_attr=weight_attr)
    
    acceptance["sampled_indices"] = indices
    acceptance["best_repair"] = best_repair
    acceptance["num_shots"] = shots
    acceptance["num_improving_samples"] = num_improving_samples
    acceptance["improving_sample_probability"] = improving_sample_probability

    return acceptance

