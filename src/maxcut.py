# Author: Amir Lorvand
"""
Max-Cut primitive functions for chapter 3.1
maximisation convention is used: C_ij(z) = w_ij * (1 - z_i z_j) / 2
where each spin z_i is either -1 or +1
"""

def validate_spin_value(spin: int) -> None:
    """
    check that a spin value is valid (either -1 or +1)
    """
    if spin not in (-1, 1):
        raise ValueError(f"Spin must be -1 or +1, got{spin}")
    
def edge_cut_value(zi:int, zj: int, weight: float=1.0) -> float:
    """
    calculate the Max-Cut contribution of one edge

    formula: C_ij = weight * (1 - zi * zj) / 2

    if zi and zj are the same, the edge is not cut
    if zi and zj are different, the edge is cut
    """
    validate_spin_value(zi)
    validate_spin_value(zj)

    return weight * (1 - zi * zj) / 2

def validate_spin_assignment(G, z: dict) -> None:
    """
    check that every node in graph G has a valid spin assignment (-1 or +1)
    z should be a dictionary like: {0: 1, 1: -1, 2: 1}
    """
    for node in G.nodes:
        if node not in z:
            raise ValueError(f"missing spin assignment for node {node}")
        
        validate_spin_value(z[node])

def cut_value(G, z: dict, weight_attr: str = "weight") -> float:
    """
    calculate the full max-cut value of graph G for assignment z

    formula: C(z) = sum over edges (i, j) of w_ij * (1 - z_i * z_j) / 2

    if an edge has no weight then weight = 1
    """
    validate_spin_assignment(G, z)

    total = 0.0

    for i, j, data in G.edges(data=True):
        weight = data.get(weight_attr, 1.0)
        total += edge_cut_value(z[i], z[j], weight)

    return total

def spin_flip(z: dict) -> dict:
    """
    return the global spin flipped assignment.

    if z_i = +1 it becomes -1
    if z_i = -1 it becomes +1

    max-cut is invariant under global spin flip: C(z) = C(-z)
    """
    flipped = {}

    for node, spin in z.items():
        validate_spin_value(spin)
        flipped[node] = -spin

    return flipped
