# Author: Amir Lorvand
"""
fixed/transferred QAOA angle utilities for chapter 3.8
this module loads source QAOA angles before effctive-scale rescaling
"""

import numpy as np 
from boundary import boundary_fields
"""
reference coefficient of an unweighted-interior ZZ term, w_ik/2 with w_ik = 1.
The effective scale is measured relative to this so that a pure unweighted
interior subproblem yields s_eff = 1 and the transferred angles are applied
as published; weighted or boundary-field deviations move s_eff away from 1.
"""
UNWEIGHTED_INTERIOR_REF = 0.5

"""
fixed/transferred QAOA angles for 3-regular Max-Cut, optimal to the tree
subgraph, from Wurtz and Lykov (2021), Table I. These are "universally good"
angles that let QAOA bypass per-instance variational optimisation on regular
Max-Cut. The cost operator there is C_hat = (1/2) sum (1 - Z_i Z_j), which
matches this project's unit-weight cut value, so gamma is directly usable
against the cost vector (subject to the relative effective-scale rescaling).
"""
FIXED_ANGLE_TABLE = {
    "wurtz_lykov_3reg_p1": {
        "gammas": [0.616],
        "betas": [0.393],
    },
    "wurtz_lykov_3reg_p2": {
        "gammas": [0.488, 0.898],
        "betas": [0.555, 0.293],
    },
    # retained for unit tests only; not a real angle source
    "test_p1": {
        "gammas": [0.5],
        "betas": [0.3],
    },
}

def load_fixed_angles(source: str, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    load fixed/source QAOA angles
    source selects the angle set
    p selects the QAOA depth
    """
    if p <= 0:
        raise ValueError("p must be positive")
    
    if source not in FIXED_ANGLE_TABLE:
        raise ValueError(f"unknown angle source: {source}")
    
    data = FIXED_ANGLE_TABLE[source]
    gammas = np.asarray(data["gammas"], dtype=float)
    betas = np.asarray(data["betas"], dtype=float)

    if len(gammas) != p or len(betas) != p:
        raise ValueError("angle source does not match requested p")
    
    return gammas, betas

def rescale_angles(gammas: np.ndarray, betas: np.ndarray, 
                   s_eff: float) -> tuple[np.ndarray, np.ndarray]:
    """
    rescale QAOA angles using effective cost scale
    gamma is divided by s_eff
    beta is left unchanged
    """
    if s_eff <= 0:
        raise ValueError("s_eff must be positive")
    
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)

    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    
    deployed_gammas = gammas / s_eff
    deployed_betas = betas.copy()

    return deployed_gammas, deployed_betas

def subproblem_scale_coefficients(G, S: list, z: dict, boundary_on=True, 
                                  weight_attr: str="weight") -> list:
    """
    collect cost coefficients used to define the effective scale, includes:
    1. interior edge weights
    2. boundary local field coefficients h_i / 2
    """
    S_set = set(S)
    coefficients = []

    # interior ZZ-type edge coefficients
    for i, j, data in G.edges(data=True):
        if i in S_set and j in S_set:
            weight = data.get(weight_attr, 1.0)
            coefficients.append(weight / 2.0)

    # boundary Z-type local field coefficients
    h = boundary_fields(G, S, z, boundary_on=boundary_on, weight_attr=weight_attr)

    for field in h:
        a_i = field / 2.0

        if a_i != 0:
            coefficients.append(a_i)

    return coefficients

def effective_scale(coefficients: list) -> float:
    """
    compute the RMS effective scale of the subproblem coefficients,
    measured relative to the unweighted-interior reference so that a pure
    unweighted interior subproblem returns 1.0 (transferred angles applied
    as published) and weighted/boundary-field deviations move it away from 1.
    """
    if len(coefficients) == 0:
        return 1.0

    coeffs = np.asarray(coefficients, dtype=float)
    rms = np.sqrt(np.mean(coeffs ** 2))

    if rms <= 0:
        return 1.0

    return float(rms / UNWEIGHTED_INTERIOR_REF)

def get_deployed_angles(G, S: list, z: dict, source: str, p: int,
                        boundary_on: bool = True, 
                        weight_attr: str="weight") -> dict:
    """
    load and rescale fixed QAOA angles for one repair subproblem
    retuens both source andgles and deployed angles   
    """
    source_gammas, source_betas = load_fixed_angles(source, p)
    coefficients = subproblem_scale_coefficients(G, S, z, boundary_on=boundary_on,
                                                 weight_attr=weight_attr)
    s_eff = effective_scale(coefficients)
    deployed_gammas, deployed_betas = rescale_angles(source_gammas, source_betas,
                                                     s_eff)
    
    return {
        "source_gammas": source_gammas,
        "source_betas": source_betas,
        "deployed_gammas": deployed_gammas,
        "deployed_betas": deployed_betas,
        "s_eff": s_eff,
        "scale_coefficients": coefficients
    }

