# Author: Amir Lorvand
import networkx as nx
import pytest

from maxcut import cut_value
from local_search import (
    flip_assignment,
    flip_gain,
    flip_gains,
    best_improving_flip,
    local_search,
    is_one_flip_local_optimum,
)


# checks that flip_gain matches full cut value recomputation
def test_flip_gain_matches_full_recomputation():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    z = {0: 1, 1: 1, 2: -1}

    old_cut = cut_value(G, z)
    z_new = flip_assignment(z, 0)
    new_cut = cut_value(G, z_new)

    assert flip_gain(G, z, 0) == pytest.approx(new_cut - old_cut)


# checks all flip gains on a simple triangle
def test_flip_gains_triangle():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    z = {0: 1, 1: 1, 2: 1}

    gains = flip_gains(G, z)

    assert gains == {0: 2.0, 1: 2.0, 2: 2.0}


# checks that steepest improvement chooses the best improving node
def test_best_improving_flip_returns_best_node():
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(0, 2, weight=1.0)

    z = {0: 1, 1: 1, 2: -1}

    # flipping node 0 gives gain 4 annd flipping node 1 gives gain 5
    assert best_improving_flip(G, z) == 1


# checks that no improving move returns None
def test_best_improving_flip_returns_none_at_local_optimum():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    z = {0: -1, 1: 1, 2: 1}

    assert best_improving_flip(G, z) is None


# checks that local search improves the initial solution
def test_local_search_improves_cut_value():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    z0 = {0: 1, 1: 1, 2: 1}

    z_star, history = local_search(G, z0)

    assert cut_value(G, z_star) >= cut_value(G, z0)
    assert len(history) > 0


# checks that local search ends at a 1flip local optimum
def test_local_search_ends_at_one_flip_local_optimum():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    z0 = {0: 1, 1: 1, 2: 1}

    z_star, history = local_search(G, z0)

    assert is_one_flip_local_optimum(G, z_star)


# checks weighted graph behaviour
def test_weighted_flip_gain():
    G = nx.Graph()
    G.add_edge("a", "b", weight=2.0)
    G.add_edge("a", "c", weight=4.0)

    z = {"a": 1, "b": 1, "c": -1}

    # gain = 2*(1*1) + 4*(1*-1) = -2
    assert flip_gain(G, z, "a") == pytest.approx(-2.0)