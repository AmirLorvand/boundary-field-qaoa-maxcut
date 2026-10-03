# Author: Amir Lorvand
"""
boundary field utilities for chapter 3.4
these function compute the local fields created by frozen outside vertices
"""

from maxcut import validate_spin_assignment, edge_cut_value, cut_value
import itertools

def boundary_fields(G, S: list, z: dict, 
                    boundary_on: bool=True, weight_attr: str="weight") -> list:
    """
    compute boundary field values for vertices inside S
    for each inside vertex i: h_i = sum over outside neighbours j of weight * z_j
    the returned list follows the same order as S
    """

    validate_spin_assignment(G, z)

    if len(S) != len(set(S)):
        raise ValueError("S must not contain duplicate vertices")
    
    for node in S:
        if node not in G.nodes:
            raise ValueError(f"Node {node} in S is not in graph")
        
    if boundary_on is False:
        return [0.0 for _ in S]
    
    S_set = set(S)
    h = []

    for i in S:
        field = 0.0
        for j in G.neighbors(i):
            if j not in S_set:
                weight = G[i][j].get(weight_attr, 1.0)
                field += weight * z[j]
        h.append(field)

    return h

def subproblem_cost(G, S: list, z_S: dict, h: list,
                    weight_attr: str='weight') -> float:
    """
    compute the boundary aware subproblem cost for one assignment over S
    this includes:
    1. interior edge cut values
    2. boundary field contribution
    z_S contain pin assignments only fro vertices inside S
    h follows the same order as S
    """
    if len(S) != len(h):
        raise ValueError("S and h mist have the same length")
    
    S_set = set(S)
    total = 0.0

    # interior edge contribution
    for i, j in G.edges:
        if i in S_set and j in S_set:
            weight = G[i][j].get(weight_attr, 1.0)
            total += edge_cut_value(z_S[i], z_S[j], weight)

    # boundary field contribution
    for index, i in enumerate(S):
        total += (h[index] / 2) * (1 - z_S[i])

    return total

def build_sub_hamiltonian(G, S: list, z: dict, boundary_on: bool=True,
                         weight_attr: str='weight') -> tuple:
    """
    build the diagonal subproblem hamiltonian over all assignments of S
    each entry in costs corresponds to one possible spin assignment over S
    assignments stores the matching spin dictionary for each cost
    """
    h = boundary_fields(G, S, z, boundary_on=boundary_on, weight_attr=weight_attr)
    costs = []
    assignments = []

    for spins in itertools.product([-1, 1], repeat=len(S)):
        z_S = dict(zip(S, spins))
        cost = subproblem_cost(G, S, z_S, h, weight_attr=weight_attr)
        assignments.append(z_S)
        costs.append(cost)

    return costs, assignments

def dropped_constant(G, S: list, z: dict, weight_attr: str="weight") -> float:
    """
    compute the constant part dropped from the boundary aware subproblem
    this includes:
    1. exterior edges where both endpoints are outside S
    2. constant parts of boundary edges
    this constant deos not depend on the assignment inside S
    """
    validate_spin_assignment(G, z)
    S_set = set(S)
    constant = 0.0

    for i, j in G.edges:
        weight = G[i][j].get(weight_attr, 1.0)

        i_inside = i in S_set
        j_inside = j in S_set 

        # exterior edge contribution is fully fixed
        if not i_inside and not j_inside:
            constant += edge_cut_value(z[i], z[j], weight)

        # boundary edge constant depends only on frozen outside spin z[j]
        elif i_inside and not j_inside:
            constant += weight * (1 - z[j]) / 2
        
        # boundary edge constant depends only on frozen outside spin z[i]
        elif j_inside and not i_inside:
            constant += weight * (1 - z[i]) / 2

    return constant

