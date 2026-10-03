# Author: Amir Lorvand
"""
statevector QAOA utilities fro chapter 3.6
this module applies:
1. the cost phase operator from the diagonal subproblem hamiltonian
2. the standard X mixer
3. repeated QAOA layers
"""

import numpy as np 

def num_qubits_from_state_length(length: int) -> int:
    """
    infer the number of qubits from the statevector length
    """
    if length <= 0:
        raise ValueError("state length must be positive")
    
    k = int(np.log2(length))

    if 2**k != length:
        raise ValueError("state length must be a power of 2")
    
    return k

def uniform_state(k: int) -> np.ndarray:
    """
    create the standard uniform QAOA initial state over k qubits
    """
    if k <= 0:
        raise ValueError("k must be positive")
    
    dim = 2**k
    return np.ones(dim, dtype=complex) / np.sqrt(dim)

def apply_phase(costs, gamma: float, state: np.ndarray) -> np.ndarray:
    """
    apply the diagnoal cost phase operator
    each basis state amplitude is multiplied by: exp(-i * gamma * cost)
    """
    costs = np.asarray(costs, dtype=float)
    state = np.asarray(state, dtype=complex)

    if len(costs) != len(state):
        raise ValueError("costs and state must have the same length")
    
    phases = np.exp(-1j * gamma * costs)

    return phases * state

def apply_mixer(beta: float, state: np.ndarray) -> np.ndarray:
    """
    apply the standard transverse-field X mixer
    the mixer is exp(-i * beta * sum_i X_i)
    becasue the X terms commute, this is equivalent to applying
    exp(-i * beta * X) to each qubit
    """
    state = np.asarray(state, dtype=complex)
    k = num_qubits_from_state_length(len(state))
    mixed_state = state.reshape([2] * k)

    single_qubit_mixer = np.array([
        [np.cos(beta), -1j * np.sin(beta)],
        [-1j * np.sin(beta), np.cos(beta)],
    ], dtype=complex)

    for qubit in range(k):
        mixed_state = np.tensordot(
            single_qubit_mixer,
            mixed_state,
            axes=([1], [qubit]),
        )
        mixed_state = np.moveaxis(mixed_state, 0, qubit)
    
    return mixed_state.reshape(-1)

def qaoa_state(costs, gammas, betas, psi0: np.ndarray) -> np.ndarray:
    """
    applyp layers of QAOA to an initial state
    costs is the diagonal subproblem hamiltonian
    gammas and betas are the QAOA angles
    psi0 is the initial state
    """
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    
    state = np.asarray(psi0, dtype=complex)

    for gamma, beta in zip(gammas, betas):
        state = apply_phase(costs, gamma, state)
        state = apply_mixer(beta, state)

    return state

def state_norm(state: np.ndarray) -> float:
    """
    compute the squared norm of a statevector
    """
    state = np.asarray(state, dtype=complex)

    return float(np.sum(np.abs(state) ** 2))

def expected_cost(costs, state: np.ndarray) -> float:
    """
    compute the expected subproblem cost of a statevector
    """
    costs = np.asarray(costs, dtype=float)
    state = np.asarray(state, dtype=complex)

    if len(costs) != len(state):
        raise ValueError("costs and state must have the same length")
    
    probabilities = np.abs(state) ** 2

    return float(np.sum(probabilities * costs))

