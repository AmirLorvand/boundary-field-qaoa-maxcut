# Author: Amir Lorvand
import itertools
import networkx as nx
import pytest

from maxcut import cut_value
from boundary import (
    boundary_fields,
    subproblem_cost,
    dropped_constant,
)


# checks the hand computed boundary field values
def test_boundary_fields_hand_example():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 4)

    S = [1, 2]

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
        4: 1,
    }

    h = boundary_fields(G, S, z, boundary_on=True)

    assert h == [0.0, 1.0]


# checks that no-boundary mode removes all boundary fields
def test_boundary_fields_zero_when_boundary_off():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 4)

    S = [1, 2]

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
        4: 1,
    }

    h = boundary_fields(G, S, z, boundary_on=False)

    assert h == [0.0, 0.0]


# checks the reconstruction identity for all assignments over S
def test_boundary_reconstruction_identity_all_assignments():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 4)

    S = [1, 2]

    z_incumbent = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
        4: 1,
    }

    h = boundary_fields(G, S, z_incumbent, boundary_on=True)
    constant = dropped_constant(G, S, z_incumbent)

    for spins in itertools.product([-1, 1], repeat=len(S)):
        z_S = dict(zip(S, spins))

        z_full = z_incumbent.copy()
        z_full.update(z_S)

        full_cut = cut_value(G, z_full)
        reconstructed = subproblem_cost(G, S, z_S, h) + constant

        assert reconstructed == pytest.approx(full_cut)

# checks reconstruction identity also works with non unit edge weights
def test_boundary_reconstruction_identity_weighted():
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(1, 2, weight=1.5)
    G.add_edge(1, 3, weight=3.0)
    G.add_edge(2, 4, weight=0.5)

    S = [1, 2]

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
        4: 1,
    }

    h = boundary_fields(G, S, z, boundary_on=True)
    constant = dropped_constant(G, S, z)

    for spins in itertools.product([-1, 1], repeat=len(S)):
        z_S = dict(zip(S, spins))

        z_full = z.copy()
        z_full.update(z_S)

        reconstructed = subproblem_cost(G, S, z_S, h) + constant
        full_cut = cut_value(G, z_full)

        assert reconstructed == pytest.approx(full_cut)