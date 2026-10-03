# Author: Amir Lorvand
import itertools
import networkx as nx 
from maxcut import edge_cut_value, cut_value, spin_flip

def brute_force_best_cut_value(G):
    """"
    try every possible spin assignment and return the best cut value,
    only for tiny graphs, used for verification
    """""
    nodes = list(G.nodes)
    best = float("-inf")

    for spins in itertools.product([-1, 1], repeat=len(nodes)):
        z = dict(zip(nodes, spins))
        value = cut_value(G, z)
        best = max(best, value)

    return best

# checks the single edge max-cut formula
def test_edge_cut_value():
    assert edge_cut_value(1, 1) == 0.0
    assert edge_cut_value(-1, -1) == 0.0
    assert edge_cut_value(1, -1) == 1.0
    assert edge_cut_value(-1, 1) == 1.0
    assert edge_cut_value(1, -1, weight=2.5) == 2.5

# checks that a triangle has maximum cut value 2
def test_triangle_best_cut_is_two():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    assert brute_force_best_cut_value(G) == 2.0

# checks that a 4 cycle has maximum cut value 4
def test_four_cycle_best_cut_is_four():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    assert brute_force_best_cut_value(G) == 4.0

# checks C(z) = C(-z)
def test_global_spin_flip_invariance():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    z = {0: 1, 1: -1, 2: 1}

    assert cut_value(G, z) == cut_value(G, spin_flip(z))

# checks weighted max-cut calculatio
def test_weighted_graph_cut_value():
    G = nx.Graph()
    G.add_edge("a", "b", weight=2.0)
    G.add_edge("b", "c", weight=3.5)
    G.add_edge("a", "c", weight=4.0)

    z = {"a": 1, "b": -1, "c": -1}

    assert cut_value(G, z) == 6.0

# checks that missing assignement are rejected
def test_missing_node_assignment_raises_error():
    G = nx.Graph()
    G.add_edge(0, 1)

    z = {0: 1}

    try:
        cut_value(G, z)
        assert False
    except ValueError:
        assert True

# checks that invalid spin values are rejected
def test_invalid_spin_value_raises_error():
    G = nx.Graph()
    G.add_edge(0, 1)

    z = {0: 1, 1: 0}

    try:
        cut_value(G, z)
        assert False
    except ValueError:
        assert True