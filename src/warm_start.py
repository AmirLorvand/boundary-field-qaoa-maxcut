# Author: Amir Lorvand
"""
warm-start utilities for chapter 3.7
these functions prepare an initial QAOA state concentrated near
the incumbent assignment over the selected neighbourhood S
"""

from maxcut import validate_spin_assignment
import numpy as np 

def restrict_assignment_to_S(G, S: list, z: dict) -> dict:
    """
    retrict a full spin assignment to the selected vertices S
    the returned dictionary follows the verices in S
    """
    validate_spin_assignment(G, z)

    if len(S) != len(set(S)):
        raise ValueError("S must be not contain duplicated vertices")
    
    for node in S:
        if node not in G.nodes:
            raise ValueError(f"Node {node} in S is not in graph")
        
    return {node: z[node] for node in S}

def spin_to_bit(spin: int) -> int:
    """
    convert spin convention to bit convention
    spin -1 maps to bit 0
    spin +1 maps to bit 1
    """
    if spin == -1:
        return 0
    
    if spin == 1:
        return 1
    
    raise ValueError(f"Spin must be -1 or +1, got {spin}")

def warm_start_probabilities(z_S: dict, S: list, epsilon: float) -> list:
    """
    compute bit-1 probabilities for the warm-start state
    if incumbent bit is 1, prob of bit 1 is 1 - epsilon
    if incumbent bit is 0, prob of bit 1 is epsilon
    """
    if epsilon < 0 or epsilon > 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")
    
    probabilities = []

    for node in S:
        spin = z_S[node]
        bit = spin_to_bit(spin)

        if bit == 1:
            probabilities.append(1.0 - epsilon)
        else:
            probabilities.append(epsilon)
    
    return probabilities

def single_qubit_warm_state(p: float) -> np.ndarray:
    """
    create a single-qubit warm-start state from bit_1 probability p
    state = ssqrt(1 - p) |0> + sqrt(p) |1>
    """
    if p < 0 or p > 1:
        raise ValueError("probability must be between 0 and 1")
    
    return np.array([
        np.sqrt(1.0 - p),
        np.sqrt(p),
    ], dtype=complex)

def tensor_product_states(states: list) -> np.ndarray:
    """
    combine single qubit states into one multi qubit statevector
    the order of states follows the order of S
    """
    if len(states) == 0:
        raise ValueError("states list must not be empty")
    
    result = states[0]

    for state in states[1:]:
        result = np.kron(result, state)
    
    return result

def warm_start_state(G, S: list, z: dict, epsilon: float) -> np.ndarray:
    """
    biuld the full warm start state over the selected vertices S
    teh state is concentrated near incumbent assignment restricted to S
    """
    z_S = restrict_assignment_to_S(G, S, z)
    probabilities = warm_start_probabilities(z_S, S, epsilon)
    states = []

    for p in probabilities:
        states.append(single_qubit_warm_state(p))

    return tensor_product_states(states)

def incumbent_bitstring(G, S: list, z: dict) -> str:
    """
    return the incumbent bitstring over S
    convention:
    spin -1 -> bit 0
    spin +1 -> bit 1
    S[0] is the first bit so the oredering matches the cost vector
    """
    z_S = restrict_assignment_to_S(G, S, z)
    bits = []

    for node in S:
        bit = spin_to_bit(z_S[node])
        bits.append(str(bit))
    
    return "".join(bits)

def bitstring_to_index(bitstring: str) -> int:
    """
    convert a bitstring to its statevector index
    example: "110" -> 6
    """
    for bit in bitstring:
        if bit not in ("0", "1"):
            raise ValueError("bitstring must contain only 0 and 1")
        
    return int(bitstring, 2)

def incumbent_basis_index(G, S: list, z: dict) -> int:
    """
    return the statevector index of the incumbent assignment over S
    """
    bitstring = incumbent_bitstring(G, S, z)

    return bitstring_to_index(bitstring)

def warmstart_overlap(state: np.ndarray, G, S: list, z: dict) -> float:
    """
    compute prob mass on the incumbent basis state
    this is used as a stalling diagnostic
    """
    state = np.asarray(state, dtype=complex)
    index = incumbent_basis_index(G, S, z)

    if index >= len(state):
        raise ValueError("incumbent index is outside the statevector length")
    
    return float(np.abs(state[index]) ** 2)

