# Author: Amir Lorvand
import networkx as nx
import pytest

from maxcut import cut_value
from repair import RepairConfig, run_single_repair_call, repair_loop


# checks RepairConfig stores the core algorithm settings
def test_repair_config():
    config = RepairConfig(
        k=3,
        p=1,
        shots=100,
        boundary_on=True,
        angle_source="test_p1",
        epsilon=0.1,
        max_rounds=5,
        seed=42,
    )

    assert config.k == 3
    assert config.p == 1
    assert config.shots == 100
    assert config.boundary_on is True
    assert config.angle_source == "test_p1"
    assert config.epsilon == pytest.approx(0.1)
    assert config.max_rounds == 5
    assert config.seed == 42
    assert config.weight_attr == "weight"


# checks one repair call can improve a poor incumbent
def test_run_single_repair_call_can_improve():
    G = nx.path_graph(5)

    z = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }

    config = RepairConfig(
        k=3,
        p=1,
        shots=100,
        boundary_on=True,
        angle_source="test_p1",
        epsilon=0.1,
        max_rounds=5,
        seed=42,
    )

    z_new, row = run_single_repair_call(
        G,
        z,
        config,
        round_id=0,
    )

    assert cut_value(G, z_new) >= cut_value(G, z)
    assert row["accepted"] is True
    assert row["new_cut"] > row["incumbent_cut"]
    assert row["accepted_improvement"] > 0


# checks one repair call returns the expected result row fields
def test_run_single_repair_call_row_fields():
    G = nx.path_graph(5)

    z = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }

    config = RepairConfig(
        k=3,
        p=1,
        shots=100,
        boundary_on=True,
        angle_source="test_p1",
        epsilon=0.1,
        max_rounds=5,
        seed=42,
    )

    z_new, row = run_single_repair_call(
        G,
        z,
        config,
        round_id=0,
    )

    required_fields = [
        "round_id",
        "seed",
        "boundary_on",
        "k",
        "p",
        "shots",
        "angle_source",
        "epsilon",
        "incumbent_cut",
        "selected_vertices",
        "num_interior_edges",
        "num_boundary_edges",
        "boundary_fields",
        "boundary_norm",
        "boundary_max_abs",
        "exact_best_cut",
        "exact_improvement",
        "s_eff",
        "source_gammas",
        "source_betas",
        "deployed_gammas",
        "deployed_betas",
        "warmstart_bitstring",
        "warmstart_overlap",
        "best_sampled_cut",
        "best_sampled_improvement",
        "num_improving_samples",
        "improving_sample_probability",
        "accepted",
        "accepted_improvement",
        "new_cut",
    ]

    for field in required_fields:
        assert field in row


# checks the full repair loop never decreases the cut value
def test_repair_loop_is_monotonic():
    G = nx.path_graph(5)

    z0 = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }

    config = RepairConfig(
        k=3,
        p=1,
        shots=100,
        boundary_on=True,
        angle_source="test_p1",
        epsilon=0.1,
        max_rounds=5,
        seed=42,
    )

    z_final, rows = repair_loop(
        G,
        z0,
        config,
    )

    assert cut_value(G, z_final) >= cut_value(G, z0)
    assert len(rows) >= 1

    for row in rows:
        assert row["new_cut"] >= row["incumbent_cut"]


# checks no boundary mode also runs through the same loop
def test_repair_loop_no_boundary_mode_runs():
    G = nx.path_graph(5)

    z0 = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }

    config = RepairConfig(
        k=3,
        p=1,
        shots=100,
        boundary_on=False,
        angle_source="test_p1",
        epsilon=0.1,
        max_rounds=5,
        seed=42,
    )

    z_final, rows = repair_loop(
        G,
        z0,
        config,
    )

    assert cut_value(G, z_final) >= cut_value(G, z0)
    assert len(rows) >= 1

    for row in rows:
        assert row["boundary_on"] is False

# checks BF and NB use the same selected neighbourhood under the same seed
def test_bf_nb_use_same_neighbourhood_with_same_seed():
    G = nx.path_graph(5)

    z = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }

    config_bf = RepairConfig(
        k=3,
        p=1,
        shots=100,
        boundary_on=True,
        angle_source="test_p1",
        epsilon=0.1,
        max_rounds=5,
        seed=42,
    )

    config_nb = RepairConfig(
        k=3,
        p=1,
        shots=100,
        boundary_on=False,
        angle_source="test_p1",
        epsilon=0.1,
        max_rounds=5,
        seed=42,
    )

    z_bf, row_bf = run_single_repair_call(
        G,
        z,
        config_bf,
        round_id=0,
    )

    z_nb, row_nb = run_single_repair_call(
        G,
        z,
        config_nb,
        round_id=0,
    )

    assert row_bf["selected_vertices"] == row_nb["selected_vertices"]
    assert row_bf["boundary_on"] is True
    assert row_nb["boundary_on"] is False

    assert row_nb["boundary_fields"] == pytest.approx([0.0, 0.0, 0.0])



# best of 6 candidate selection tests

import networkx as nx
from repair import select_best_neighbourhood, run_single_repair_call
from local_search import local_search


def _cfg(boundary_on=True, num_candidates=6, seed=42, k=12):
    return RepairConfig(
        k=k, p=1, shots=100, boundary_on=boundary_on,
        angle_source="wurtz_lykov_3reg_p1", epsilon=0.1,
        max_rounds=5, seed=seed, num_candidates=num_candidates,
    )


def test_best_of_selects_positive_headroom_when_available():
    G = nx.random_regular_graph(3, 100, seed=7)
    z_star, _ = local_search(G, {v: 1 for v in G.nodes}, seed=7)
    sel = select_best_neighbourhood(G, z_star, _cfg(), seed=123)
    if any(c > 0 for c in sel["candidate_improvements"]):
        assert sel["headroom_positive"] is True
        assert sel["candidate_improvements"][sel["selected_index"]] == max(sel["candidate_improvements"])
        assert sel["candidate_improvements"][sel["selected_index"]] > 0


def test_best_of_fallback_when_no_positive():
    G = nx.empty_graph(20)
    z = {v: 1 for v in G.nodes}
    sel = select_best_neighbourhood(G, z, _cfg(k=5), seed=1)
    assert sel["headroom_positive"] is False
    assert all(c <= 0 for c in sel["candidate_improvements"])
    assert len(sel["selected_S"]) == 5


def test_best_of_bf_nb_same_neighbourhood():
    G = nx.random_regular_graph(3, 100, seed=11)
    z_star, _ = local_search(G, {v: 1 for v in G.nodes}, seed=11)
    _, row_bf = run_single_repair_call(G, z_star, _cfg(boundary_on=True), round_id=0)
    _, row_nb = run_single_repair_call(G, z_star, _cfg(boundary_on=False), round_id=0)
    assert row_bf["selected_vertices"] == row_nb["selected_vertices"]
    assert row_bf["selected_candidate_index"] == row_nb["selected_candidate_index"]
    assert row_bf["headroom_positive"] == row_nb["headroom_positive"]


def test_best_of_row_fields_present():
    G = nx.random_regular_graph(3, 100, seed=3)
    z_star, _ = local_search(G, {v: 1 for v in G.nodes}, seed=3)
    _, row = run_single_repair_call(G, z_star, _cfg(), round_id=0)
    for f in ("num_candidates", "candidate_exact_improvements",
              "selected_candidate_index", "headroom_positive"):
        assert f in row
    assert row["num_candidates"] == 6
    assert len(row["candidate_exact_improvements"]) == row["num_candidates"]
    assert 0 <= row["selected_candidate_index"] < row["num_candidates"]


def test_best_of_loop_monotonic():
    from repair import repair_loop
    G = nx.random_regular_graph(3, 100, seed=5)
    z0 = {v: 1 for v in G.nodes}
    z_final, rows = repair_loop(G, z0, _cfg())
    assert cut_value(G, z_final) >= cut_value(G, z0)
    for row in rows:
        assert row["new_cut"] >= row["incumbent_cut"]
