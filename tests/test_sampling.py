# Author: Amir Lorvand
import networkx as nx
import numpy as np
import pytest

from boundary import build_sub_hamiltonian
from qaoa import uniform_state
from sampling import (
    state_probabilities,
    sample_state_indices,
    decode_sampled_indices,
    evaluate_sampled_assignments,
    best_sampled_repair,
    accept_repair_if_improves,
    sample_and_accept_repair,
)

# checks statevector amplitudes are converted to probabilities
def test_state_probabilities():
    state = np.array([1.0, 0.0], dtype=complex)

    probabilities = state_probabilities(state)

    assert np.allclose(probabilities, [1.0, 0.0])


# checks probabilities are normalised even if the state is not normalised
def test_state_probabilities_normalises():
    state = np.array([2.0, 0.0], dtype=complex)

    probabilities = state_probabilities(state)

    assert np.allclose(probabilities, [1.0, 0.0])


# checks empty states are rejected
def test_state_probabilities_rejects_empty_state():
    with pytest.raises(ValueError):
        state_probabilities(np.array([], dtype=complex))


# checks sampled indices are valid statevector indices
def test_sample_state_indices_are_valid():
    state = uniform_state(2)

    indices = sample_state_indices(
        state,
        shots=20,
        seed=42,
    )

    assert len(indices) == 20
    assert np.all(indices >= 0)
    assert np.all(indices < 4)


# checks zero or negative shot counts are rejected
def test_sample_state_indices_rejects_invalid_shots():
    state = uniform_state(2)

    with pytest.raises(ValueError):
        sample_state_indices(state, shots=0)


# checks sampled indices decode through the assignments list
def test_decode_sampled_indices():
    G = nx.path_graph(3)

    S = [0, 1]

    z = {
        0: 1,
        1: 1,
        2: 1,
    }

    costs, assignments = build_sub_hamiltonian(
        G,
        S,
        z,
        boundary_on=True,
    )

    decoded = decode_sampled_indices(
        [0, 1, 2, 3],
        assignments,
    )

    assert decoded == [
        {0: -1, 1: -1},
        {0: -1, 1: 1},
        {0: 1, 1: -1},
        {0: 1, 1: 1},
    ]


# checks invalid sampled indices are rejected
def test_decode_sampled_indices_rejects_invalid_index():
    assignments = [
        {0: -1},
        {0: 1},
    ]

    with pytest.raises(ValueError):
        decode_sampled_indices([2], assignments)

# checks sampled assignments are evaluated on the full graph
def test_evaluate_sampled_assignments():
    G = nx.path_graph(3)

    S = [0, 1]

    z = {
        0: 1,
        1: 1,
        2: 1,
    }

    sampled_assignments = [
        {0: -1, 1: -1},
        {0: -1, 1: 1},
        {0: 1, 1: -1},
        {0: 1, 1: 1},
    ]

    evaluations = evaluate_sampled_assignments(
        G,
        S,
        z,
        sampled_assignments,
    )

    cuts = [item["cut"] for item in evaluations]
    improvements = [item["improvement"] for item in evaluations]

    assert cuts == pytest.approx([1.0, 1.0, 2.0, 0.0])
    assert improvements == pytest.approx([1.0, 1.0, 2.0, 0.0])


# checks the best sampled repair is selected by full cut value
def test_best_sampled_repair():
    evaluations = [
        {
            "z_S": {0: -1},
            "full_assignment": {0: -1, 1: 1},
            "cut": 1.0,
            "improvement": 1.0,
        },
        {
            "z_S": {0: 1},
            "full_assignment": {0: 1, 1: -1},
            "cut": 2.0,
            "improvement": 2.0,
        },
    ]

    best = best_sampled_repair(evaluations)

    assert best["cut"] == pytest.approx(2.0)
    assert best["improvement"] == pytest.approx(2.0)


# checks empty evaluation lists are rejected
def test_best_sampled_repair_rejects_empty_list():
    with pytest.raises(ValueError):
        best_sampled_repair([])


# checks improving repairs are accepted
def test_accept_repair_if_improves_accepts_improvement():
    G = nx.path_graph(3)

    z = {
        0: 1,
        1: 1,
        2: 1,
    }

    best_repair = {
        "z_S": {0: 1, 1: -1},
        "full_assignment": {0: 1, 1: -1, 2: 1},
        "cut": 2.0,
        "improvement": 2.0,
    }

    result = accept_repair_if_improves(
        G,
        z,
        best_repair,
    )

    assert result["accepted"] is True
    assert result["new_assignment"] == {0: 1, 1: -1, 2: 1}
    assert result["incumbent_cut"] == pytest.approx(0.0)
    assert result["candidate_cut"] == pytest.approx(2.0)
    assert result["new_cut"] == pytest.approx(2.0)
    assert result["improvement"] == pytest.approx(2.0)


# checks non improving repairs are rejected
def test_accept_repair_if_improves_rejects_non_improvement():
    G = nx.path_graph(3)

    z = {
        0: 1,
        1: -1,
        2: 1,
    }

    best_repair = {
        "z_S": {0: 1, 1: -1},
        "full_assignment": {0: 1, 1: -1, 2: 1},
        "cut": 2.0,
        "improvement": 0.0,
    }

    result = accept_repair_if_improves(
        G,
        z,
        best_repair,
    )

    assert result["accepted"] is False
    assert result["new_assignment"] == z
    assert result["incumbent_cut"] == pytest.approx(2.0)
    assert result["candidate_cut"] == pytest.approx(2.0)
    assert result["new_cut"] == pytest.approx(2.0)
    assert result["improvement"] == pytest.approx(0.0)


# checks the full sample,decode,evaluate, and accept pipeline
def test_sample_and_accept_repair_end_to_end():
    G = nx.path_graph(3)

    S = [0, 1]

    z = {
        0: 1,
        1: 1,
        2: 1,
    }

    costs, assignments = build_sub_hamiltonian(
        G,
        S,
        z,
        boundary_on=True,
    )

    state = uniform_state(k=len(S))

    result = sample_and_accept_repair(
        G,
        S,
        z,
        state,
        assignments,
        shots=100,
        seed=42,
    )

    assert result["accepted"] is True
    assert result["new_cut"] == pytest.approx(2.0)
    assert result["improvement"] == pytest.approx(2.0)
    assert result["num_shots"] == 100
    assert result["num_improving_samples"] > 0
    assert result["improving_sample_probability"] > 0.0
    assert result["improving_sample_probability"] <= 1.0

