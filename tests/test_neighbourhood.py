# Author: Amir Lorvand
import networkx as nx

from neighbourhood import (
    select_neighbourhood,
    generate_candidate_neighbourhoods,
    partition_edges,
)


# checks that select_neighbourhood returns exactly k nodes
def test_select_neighbourhood_returns_k_nodes():
    G = nx.cycle_graph(8)
    z = {node: 1 for node in G.nodes}

    S = select_neighbourhood(G, z, k=3, seed=42)

    assert len(S) == 3


# checks that the same seed gives the same selected neighbourhood
def test_select_neighbourhood_is_reproducible():
    G = nx.cycle_graph(8)
    z = {node: 1 for node in G.nodes}

    S1 = select_neighbourhood(G, z, k=3, seed=42)
    S2 = select_neighbourhood(G, z, k=3, seed=42)

    assert S1 == S2


# checks that one selected neighbourhood does not contain duplicate nodes
def test_select_neighbourhood_has_no_duplicate_nodes():
    G = nx.cycle_graph(8)
    z = {node: 1 for node in G.nodes}

    S = select_neighbourhood(G, z, k=4, seed=42)

    assert len(S) == len(set(S))


# checks that generate_candidate_neighbourhoods returns the requested number of candidates
def test_generate_candidate_neighbourhoods_returns_requested_number():
    G = nx.cycle_graph(8)
    z = {node: 1 for node in G.nodes}

    candidates = generate_candidate_neighbourhoods(
        G,
        z,
        k=3,
        num_candidates=5,
        seed=42,
    )

    assert len(candidates) == 5


# checks that generated candidate neighbourhoods are unique as sets
def test_generate_candidate_neighbourhoods_are_unique_as_sets():
    G = nx.cycle_graph(8)
    z = {node: 1 for node in G.nodes}

    candidates = generate_candidate_neighbourhoods(
        G,
        z,
        k=3,
        num_candidates=5,
        seed=42,
    )

    keys = [frozenset(S) for S in candidates]

    assert len(keys) == len(set(keys))


# checks that every generated candidate has exactly k nodes
def test_generate_candidate_neighbourhoods_each_candidate_has_k_nodes():
    G = nx.cycle_graph(8)
    z = {node: 1 for node in G.nodes}

    candidates = generate_candidate_neighbourhoods(
        G,
        z,
        k=3,
        num_candidates=5,
        seed=42,
    )

    for S in candidates:
        assert len(S) == 3


# checks that partition_edges correctly separates inside, boundary and outside edges
def test_partition_edges():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 4),
    ])

    S = [0, 1, 2]

    parts = partition_edges(G, S)

    assert set(parts["inside_edges"]) == {(0, 1), (1, 2)}
    assert set(parts["boundary_edges"]) == {(2, 3), (0, 4)}
    assert set(parts["outside_edges"]) == {(3, 4)}

# checks that a selected neighbourhood has at least one boundary edge
# this is important because boundary field repair needs boundary edges to create non-zero boundary terms
def test_selected_neighbourhood_has_boundary_edges():
    G = nx.cycle_graph(8)
    z = {node: 1 for node in G.nodes}

    S = select_neighbourhood(G, z, k=3, seed=42)
    parts = partition_edges(G, S)

    assert len(parts["boundary_edges"]) > 0