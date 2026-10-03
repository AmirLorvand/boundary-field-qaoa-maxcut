# Author: Amir Lorvand
"""
neighbourhood selection utilities for chapter 3.3
these functions choose the active subgraph S that will later be repaired
vertices outside S are treated as frozen in later section
"""
import random
from maxcut import validate_spin_assignment

def select_neighbourhood(G, z: dict, k:int, seed:int | None = None) -> list:
    """
    select a connected-ish neighbourhood of k vertices
    the procedure:
    1. pick a random start vertex
    2. expand through graph neighbours until k vertices are selected
    3. return the selected vertices as an ordered list
    the seed make the selection reproducible
    """
    if k <= 0:
        raise ValueError("k must be positive")
    
    if k > G.number_of_nodes():
        raise ValueError(" k cannot be larger thn the number of graph nodes")
    
    validate_spin_assignment(G, z)
        
    rng = random.Random(seed)

    nodes = list(G.nodes)
    start_node = rng.choice(nodes)

    selected = [start_node]
    selected_set = {start_node}

    frontier = list(G.neighbors(start_node))
    rng.shuffle(frontier)

    while len(selected) < k:
        if frontier:
            candidate = frontier.pop(0)
        else:
            # handle disconnected graphs
            remaining = [node for node in nodes if node not in selected_set]
            candidate = rng.choice(remaining)

        if candidate in selected_set:
            continue

        selected.append(candidate)
        selected_set.add(candidate)

        new_neighbours = list(G.neighbors(candidate))
        rng.shuffle(new_neighbours)

        for neighbour in new_neighbours:
            if neighbour not in selected_set:
                frontier.append(neighbour)

    return selected

def generate_candidate_neighbourhoods(G, z: dict, k: int, num_candidates: int,
                                      seed: int | None = None) -> list:
    
    """
    generate multiple candidate neighbourhoods
    each candidate is selected using select_neighbourhood
    different seeds are used so candidates are not all identical
    duplicate neighbourhoods are aavoided
    """
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")

    rng = random.Random(seed)

    candidates = []
    seen = set()

    max_attempts = num_candidates * 20
    attempts = 0

    while len(candidates) < num_candidates and attempts < max_attempts:
        candidate_seed = rng.randint(0, 10**9)
        S = select_neighbourhood(G, z, k, seed=candidate_seed)

        key = frozenset(S)

        if key not in seen:
            candidates.append(S)
            seen.add(key)

        attempts += 1

    return candidates

def partition_edges(G, S: list) -> dict:
    """
    split graph edges into inside, boundary, and outside edges
    inside edge: both endpoints are inside S
    boundary edge: one endpoint is inside S and one endpoint is outside S
    outside edge: both edpoints are outside S
    """
    S_set = set(S)

    inside_edges = []
    boundary_edges = []
    outside_edges = []

    for i, j in G.edges:
        i_inside = i in S_set
        j_inside = j in S_set

        if i_inside and j_inside:
            inside_edges.append((i, j))
        elif i_inside or j_inside:
            boundary_edges.append((i, j))
        else:
            outside_edges.append((i, j))

    return {
        "inside_edges": inside_edges,
        "boundary_edges": boundary_edges,
        "outside_edges": outside_edges
    }
