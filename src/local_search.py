# Author: Amir Lorvand
"""
classical 1-flip local-search utilities for max-cut.
these functions implement the basic move operation used in section 3.2.
"""

from maxcut import cut_value, validate_spin_assignment, validate_spin_value

def flip_assignment(z: dict, node) -> dict:
    """
    return a new assignment where one node is flipped
    """
    if node not in z:
        raise ValueError(f"Node {node} is missing from assignment")
    
    validate_spin_value(z[node])
    z_new = z.copy()
    z_new[node] = -z_new[node]

    return z_new

def flip_gain(G, z: dict, node, weight_attr: str="weight") -> float:
    """
    compute the cut improvement from flipping one node
    formula: delta_i = sum over neighbours j of weight * z_i * z_j
    positive gain means theflip improves the cut.
    """
    validate_spin_assignment(G, z)

    if node not in G.nodes:
        raise ValueError(f"Node {node} is not in graph")
    
    gain = 0.0

    for neighbour in G.neighbors(node):
        weight = G[node][neighbour].get(weight_attr, 1.0)
        gain += weight * z[node] * z[neighbour]

    return gain

def flip_gains(G, z: dict, weight_attr: str='weight') -> dict:
    """
    compute flip gains for every node in the graph.
    """
    validate_spin_assignment(G, z)

    gains = {}

    for node in G.nodes:
        gains[node] = flip_gain(G, z, node, weight_attr)

    return gains

def best_improving_flip(G, z:dict, weight_attr: str="weight"):
    """
    return the node with the largest positive flip gain
    if no improving flip exsits, return none
    """
    gains = flip_gains(G, z, weight_attr)

    best_node = None
    best_gain = 0.0

    for node, gain in gains.items():
        if gain > best_gain:
            best_node = node
            best_gain = gain
    
    return best_node

def local_search(G, z0: dict, seed=None, weight_attr: str="weight"):
    """
    run steepest-improvement 1-flip local search
    the algorithm repeatedly flips the node with largest postive gain
    it stops when no single flip improves the cut
    """
    validate_spin_assignment(G, z0)

    z_current = z0.copy()
    history = []

    step = 0

    while True:
        current_cut = cut_value(G, z_current, weight_attr)
        node_to_flip = best_improving_flip(G, z_current, weight_attr)

        if node_to_flip is None:
            break

        gain = flip_gain(G, z_current, node_to_flip, weight_attr)
        z_current = flip_assignment(z_current, node_to_flip)
        new_cut = cut_value(G, z_current, weight_attr)

        history.append({
            "step": step,
            "flipped_node": node_to_flip,
            "gain": gain,
            "cut_before": current_cut,
            "cut_after": new_cut
        })

        step += 1
    
    return z_current, history

def is_one_flip_local_optimum(G, z: dict, weight_attr: str="weight") -> bool:
    """
    check whether no single-node flip can improve the cut
    """
    gains = flip_gains(G, z, weight_attr)

    for gain in gains.values():
        if gain > 0:
            return False
        
    return True
