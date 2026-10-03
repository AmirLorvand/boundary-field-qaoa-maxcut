# Author: Amir Lorvand
"""
exact subgraph repair utilities for chapter 3.5
these functions brute force all ssignments inside S while keeping vertices
outside S frozen
"""
import itertools
from maxcut import validate_spin_assignment, cut_value

def bruteforce_assignments(S: list):
    """
    generate every possible spin asignment over the selected vertices S.
    the order follows S exactly, so this must stay consistent with 
    build_sub_hamiltonian in boundary.py
    """
    for spins in itertools.product([-1, 1], repeat=len(S)):
        yield dict(zip(S, spins))

def insert_subassignment(z: dict, z_S: dict) -> dict:
    """
    create a full assignment by replacing the spins inside S
    z is the current full assignment
    z_S is the proposed assignment only over S
    """
    z_new = z.copy()
    z_new.update(z_S)

    return z_new

def exact_repair(G, S: list, z: dict, weight_attr: str='weight') -> dict:
    """
    find the best possible assignment over S by brute force
    vertices outside S remain frozen according to z
    """
    validate_spin_assignment(G, z)

    incumbent_cut = cut_value(G, z, weight_attr)

    best_z_S = {node: z[node] for node in S}
    best_full_assignment = z.copy()
    best_cut = incumbent_cut

    for z_S in bruteforce_assignments(S):
        z_candidate = insert_subassignment(z, z_S)
        candidate_cut = cut_value(G, z_candidate, weight_attr)

        if candidate_cut > best_cut:
            best_z_S = z_S
            best_full_assignment = z_candidate
            best_cut = candidate_cut

    exact_improvement = best_cut - incumbent_cut

    return {
        "best_z_S": best_z_S,
        "best_full_assignment": best_full_assignment,
        "incumbent_cut": incumbent_cut,
        "best_cut": best_cut,
        "exact_improvement": exact_improvement,
    }
