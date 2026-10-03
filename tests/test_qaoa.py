# Author: Amir Lorvand
import numpy as np
import pytest
import networkx as nx
from boundary import build_sub_hamiltonian

from qaoa import (
    uniform_state,
    apply_phase,
    apply_mixer,
    qaoa_state,
    state_norm,
    expected_cost,
)


# checks that the uniform initial state has norm 1
def test_uniform_state_has_norm_one():
    psi = uniform_state(k=3)

    assert state_norm(psi) == pytest.approx(1.0)


# checks that the phase operator preserves state norm
def test_apply_phase_preserves_norm():
    costs = [0.0, 1.0, 1.0, 2.0]
    psi = uniform_state(k=2)

    psi_after = apply_phase(costs, gamma=0.5, state=psi)

    assert state_norm(psi_after) == pytest.approx(1.0)


# checks that the X mixer preserves state norm
def test_apply_mixer_preserves_norm():
    psi = uniform_state(k=2)

    psi_after = apply_mixer(beta=0.3, state=psi)

    assert state_norm(psi_after) == pytest.approx(1.0)


# checks that zero QAOA angles leave the state unchanged
def test_qaoa_zero_angles_identity():
    costs = [0.0, 1.0, 1.0, 2.0]
    psi0 = uniform_state(k=2)

    psi_final = qaoa_state(
        costs,
        gammas=[0.0],
        betas=[0.0],
        psi0=psi0,
    )

    assert np.allclose(psi_final, psi0)


# checks expected cost on a hand computable uniform state
def test_expected_cost_uniform_state():
    costs = [0.0, 1.0, 1.0, 2.0]
    psi = uniform_state(k=2)

    # uniform probabilities are 1/4 each:
    # expected cost = (0 + 1 + 1 + 2) / 4 = 1
    assert expected_cost(costs, psi) == pytest.approx(1.0)


# checks that qaoa_state rejects mismatched angle lists
def test_qaoa_rejects_mismatched_angles():
    costs = [0.0, 1.0, 1.0, 2.0]
    psi0 = uniform_state(k=2)

    with pytest.raises(ValueError):
        qaoa_state(
            costs,
            gammas=[0.5, 0.6],
            betas=[0.3],
            psi0=psi0,
        )

# checks QAOA works with a real boundary aware subproblem Hamiltonian
def test_qaoa_with_boundary_hamiltonian_preserves_norm_and_expected_cost():
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

    psi0 = uniform_state(k=len(S))

    psi_final = qaoa_state(
        costs,
        gammas=[0.5],
        betas=[0.3],
        psi0=psi0,
    )

    assert state_norm(psi_final) == pytest.approx(1.0)

    value = expected_cost(costs, psi_final)

    assert value >= min(costs)
    assert value <= max(costs)