# Author: Amir Lorvand
import networkx as nx
import pytest 
from boundary import build_sub_hamiltonian, dropped_constant

from maxcut import cut_value
from exact_repair import (
    bruteforce_assignments,
    insert_subassignment,
    exact_repair,
)


# checks that all 2^k assignments are generated
def test_bruteforce_assignments_count():
    S = [1, 2, 3]

    assignments = list(bruteforce_assignments(S))

    assert len(assignments) == 8


# checks that subassignment insertion only changes vertices in S
def test_insert_subassignment():
    z = {0: 1, 1: -1, 2: 1, 3: -1}
    z_S = {1: 1, 2: -1}

    z_new = insert_subassignment(z, z_S)

    assert z_new == {0: 1, 1: 1, 2: -1, 3: -1}
    assert z == {0: 1, 1: -1, 2: 1, 3: -1}


# checks that exact repair can find an improving assignment
def test_exact_repair_finds_improvement():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ])

    z = {
        0: 1,
        1: 1,
        2: 1,
        3: -1,
    }

    S = [1, 2]

    result = exact_repair(G, S, z)

    assert result["incumbent_cut"] == 2.0
    assert result["best_cut"] == 4.0
    assert result["exact_improvement"] == 2.0


# checks that exact repair never returns a worse cut
def test_exact_repair_improvement_is_nonnegative():
    G = nx.cycle_graph(4)

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
    }

    S = [1, 2]

    result = exact_repair(G, S, z)

    assert result["exact_improvement"] >= 0
    assert result["best_cut"] >= result["incumbent_cut"]


# checks that the returned best assignment matches full cut evaluation
def test_exact_repair_best_assignment_matches_cut_value():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ])

    z = {
        0: 1,
        1: 1,
        2: 1,
        3: -1,
    }

    S = [1, 2]

    result = exact_repair(G, S, z)

    assert cut_value(G, result["best_full_assignment"]) == result["best_cut"]

# checks exact repair agrees with the boundary aware subHamiltonian
def test_exact_repair_matches_sub_hamiltonian():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ])

    z = {
        0: 1,
        1: 1,
        2: 1,
        3: -1,
    }

    S = [1, 2]

    result = exact_repair(G, S, z)

    costs, assignments = build_sub_hamiltonian(G, S, z, boundary_on=True)
    constant = dropped_constant(G, S, z)

    best_reconstructed_cut = max(cost + constant for cost in costs)

    assert best_reconstructed_cut == pytest.approx(result["best_cut"])