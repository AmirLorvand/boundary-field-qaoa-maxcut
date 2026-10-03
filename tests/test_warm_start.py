# Author: Amir Lorvand
import numpy as np
import networkx as nx
import pytest
from boundary import build_sub_hamiltonian
from qaoa import state_norm, qaoa_state
from warm_start import (
    spin_to_bit,
    warm_start_probabilities,
    warm_start_state,
    incumbent_bitstring,
    incumbent_basis_index,
    warmstart_overlap,
)

# checks the spin to bit convention used throughout the project
def test_spin_to_bit_convention():
    assert spin_to_bit(1) == 1
    assert spin_to_bit(-1) == 0

# checks invalid spin values are rejected
def test_spin_to_bit_rejects_invalid_spin():
    with pytest.raises(ValueError):
        spin_to_bit(0)

# checks warm start probabilities follow the incumbent assignment
def test_warm_start_probabilities():
    S = [1, 3, 4]

    z_S = {
        1: -1,
        3: -1,
        4: 1,
    }

    probs = warm_start_probabilities(z_S, S, epsilon=0.1)

    assert probs == pytest.approx([0.1, 0.1, 0.9])

# checks the warm start state has the correct length and norm
def test_warm_start_state_length_and_norm():
    G = nx.path_graph(5)

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
        4: 1,
    }

    S = [1, 3, 4]

    psi0 = warm_start_state(G, S, z, epsilon=0.1)

    assert len(psi0) == 8
    assert state_norm(psi0) == pytest.approx(1.0)

# checks the incumbent bitstring follows the ordering of S
def test_incumbent_bitstring_and_index():
    G = nx.path_graph(5)

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
        4: 1,
    }

    S = [1, 3, 4]

    assert incumbent_bitstring(G, S, z) == "001"
    assert incumbent_basis_index(G, S, z) == 1

# checks warm start overlap equals the expected incumbent probability
def test_warmstart_overlap_initial_state():
    G = nx.path_graph(5)

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
        4: 1,
    }

    S = [1, 3, 4]

    psi0 = warm_start_state(G, S, z, epsilon=0.1)

    assert warmstart_overlap(psi0, G, S, z) == pytest.approx(0.9 ** 3)

# checks that warm start state works inside the QAOA engine
def test_warm_start_state_with_qaoa_engine():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
    ])

    S = [1, 2]

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
        4: 1,
    }

    costs, assignments = build_sub_hamiltonian(G, S, z, boundary_on=True)

    psi0 = warm_start_state(G, S, z, epsilon=0.1)

    psi_final = qaoa_state(
        costs,
        gammas=[0.5],
        betas=[0.3],
        psi0=psi0,
    )

    overlap = warmstart_overlap(psi_final, G, S, z)

    assert overlap >= 0.0
    assert overlap <= 1.0

# checks that the warm start peak matches the cost vector assignment index
def test_warmstart_peak_matches_cost_vector_index():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
    ])

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
        4: 1,
    }

    S = [1, 2]

    costs, assignments = build_sub_hamiltonian(G, S, z, boundary_on=True)

    for i, z_S in enumerate(assignments):
        z_full = z.copy()
        z_full.update(z_S)

        psi0 = warm_start_state(G, S, z_full, epsilon=0.01)

        peak_index = int(np.argmax(np.abs(psi0) ** 2))

        assert peak_index == i
        assert incumbent_basis_index(G, S, z_full) == i

