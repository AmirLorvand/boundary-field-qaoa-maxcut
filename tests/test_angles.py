# Author: Amir Lorvand
import networkx as nx
import numpy as np
import pytest

from angles import (
    load_fixed_angles,
    rescale_angles,
    subproblem_scale_coefficients,
    effective_scale,
    get_deployed_angles,
)
from boundary import build_sub_hamiltonian
from qaoa import uniform_state, qaoa_state, state_norm


# checks fixed/source angles load correctly
def test_load_fixed_angles():
    gammas, betas = load_fixed_angles("test_p1", p=1)

    assert np.allclose(gammas, [0.5])
    assert np.allclose(betas, [0.3])


# checks unknown angle sources are rejected
def test_load_fixed_angles_rejects_unknown_source():
    with pytest.raises(ValueError):
        load_fixed_angles("unknown_source", p=1)


# checks gamma is rescaled and beta is unchanged
def test_rescale_angles():
    gammas = np.array([0.5])
    betas = np.array([0.3])

    deployed_gammas, deployed_betas = rescale_angles(
        gammas,
        betas,
        s_eff=2.0,
    )

    assert np.allclose(deployed_gammas, [0.25])
    assert np.allclose(deployed_betas, [0.3])


# checks effective scale uses RMS coefficient scale
def test_effective_scale():
    scale = effective_scale([1.0, 0.5])

    assert scale == pytest.approx(1.5811388300841898)


# checks empty coefficient list safely returns scale 1
def test_effective_scale_empty_coefficients():
    assert effective_scale([]) == pytest.approx(1.0)


# checks scaling all coefficients by a constant scales s_eff correctly
def test_effective_scale_scales_with_coefficients():
    base_scale = effective_scale([1.0, 0.5])
    scaled_scale = effective_scale([2.0, 1.0])

    assert scaled_scale == pytest.approx(2.0 * base_scale)


# checks larger effective scale reduces deployed gamma
def test_larger_effective_scale_reduces_gamma():
    gammas = np.array([0.5])
    betas = np.array([0.3])

    deployed_gammas_1, deployed_betas_1 = rescale_angles(
        gammas,
        betas,
        s_eff=1.0,
    )

    deployed_gammas_2, deployed_betas_2 = rescale_angles(
        gammas,
        betas,
        s_eff=2.0,
    )

    assert deployed_gammas_2[0] == pytest.approx(deployed_gammas_1[0] / 2.0)
    assert deployed_betas_2[0] == pytest.approx(deployed_betas_1[0])


# checks coefficient extraction includes interior operator coefficient and boundary field
def test_subproblem_scale_coefficients_boundary_on():
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

    coefficients = subproblem_scale_coefficients(
        G,
        S,
        z,
        boundary_on=True,
    )

    assert coefficients == pytest.approx([0.5, 0.5])


# checks no-boundary mode removes boundary field coefficients
def test_subproblem_scale_coefficients_boundary_off():
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

    coefficients = subproblem_scale_coefficients(
        G,
        S,
        z,
        boundary_on=False,
    )

    assert coefficients == pytest.approx([0.5])


# checks deployed angle pipeline works end to end
def test_get_deployed_angles():
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

    result = get_deployed_angles(
        G,
        S,
        z,
        source="test_p1",
        p=1,
        boundary_on=True,
    )

    assert np.allclose(result["source_gammas"], [0.5])
    assert np.allclose(result["source_betas"], [0.3])
    assert result["s_eff"] == pytest.approx(1.0)
    assert np.allclose(result["deployed_gammas"], [0.5])
    assert np.allclose(result["deployed_betas"], [0.3])
    assert result["scale_coefficients"] == pytest.approx([0.5, 0.5])


# checks deterministic output for the same subproblem
def test_get_deployed_angles_is_deterministic():
    G = nx.path_graph(4)

    S = [1, 2]

    z = {
        0: 1,
        1: -1,
        2: 1,
        3: -1,
    }

    result_1 = get_deployed_angles(
        G,
        S,
        z,
        source="test_p1",
        p=1,
        boundary_on=True,
    )

    result_2 = get_deployed_angles(
        G,
        S,
        z,
        source="test_p1",
        p=1,
        boundary_on=True,
    )

    assert np.allclose(result_1["deployed_gammas"], result_2["deployed_gammas"])
    assert np.allclose(result_1["deployed_betas"], result_2["deployed_betas"])
    assert result_1["s_eff"] == pytest.approx(result_2["s_eff"])


# checks deployed angles work inside the QAOA statevector engine
def test_deployed_angles_work_with_qaoa_engine():
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

    angle_data = get_deployed_angles(
        G,
        S,
        z,
        source="test_p1",
        p=1,
        boundary_on=True,
    )

    psi0 = uniform_state(k=len(S))

    psi_final = qaoa_state(
        costs,
        gammas=angle_data["deployed_gammas"],
        betas=angle_data["deployed_betas"],
        psi0=psi0,
    )

    assert state_norm(psi_final) == pytest.approx(1.0)

# smoke test: real Wurtz-Lykov p=1 angles produce a sensible improving
# distribution on a small 3-regular subproblem with known headroom
def test_wurtz_lykov_p1_angles_improve_over_uniform():
    import networkx as nx
    from boundary import build_sub_hamiltonian
    from qaoa import qaoa_state, uniform_state
    from sampling import state_probabilities

    G = nx.complete_graph(4)          # 3-regular
    S = [0, 1, 2, 3]
    z = {0: 1, 1: 1, 2: 1, 3: 1}      # cut 0, max cut 4 -> clear headroom

    result = get_deployed_angles(
        G, S, z, source="wurtz_lykov_3reg_p1", p=1, boundary_on=False,
    )
    # unweighted interior -> relative scale is 1, angles applied as published
    assert result["s_eff"] == pytest.approx(1.0)
    assert np.allclose(result["deployed_gammas"], [0.616])
    assert np.allclose(result["deployed_betas"], [0.393])

    costs, _ = build_sub_hamiltonian(G, S, z, boundary_on=False)
    psi = qaoa_state(costs, result["deployed_gammas"], result["deployed_betas"],
                     uniform_state(len(S)))
    p = state_probabilities(psi)
    best = max(costs)
    p_opt = sum(p[i] for i, c in enumerate(costs) if abs(c - best) < 1e-9)

    uniform_p_opt = sum(1 for c in costs if abs(c - best) < 1e-9) / len(costs)
    # WL angles should concentrate clearly above the uniform baseline
    assert p_opt > 1.5 * uniform_p_opt
