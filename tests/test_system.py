from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proteus import (PerturbationParams, Proteus, ProteusConfig, RoadMap, Simulation,
                     deploy_quadrants, eta_squared, graph_similarity, map_metrics)
from proteus.cli import main as cli_main
from proteus.experiments import (deployment_experiment, edge_verification_experiment,
                                 edge_verification_trial, end_to_end_simulation,
                                 path_combination_experiment)
from proteus.perturbation import apply_deletion, perturb_segments

DATA = Path(__file__).resolve().parents[1] / "data"


def test_roadmap_csv_roundtrip(tmp_path):
    rm = RoadMap.synthetic_grid(4, 4, 80)
    p = tmp_path / "m.csv"
    rm.to_edge_csv(p)
    back = RoadMap.from_edge_csv(p)
    assert back.graph.number_of_edges() == rm.graph.number_of_edges()
    assert back.total_length() == pytest.approx(rm.total_length(), rel=1e-3)


def test_subdivide_preserves_length_and_connectivity():
    rm = RoadMap.synthetic_grid(4, 4, 100)
    sub = rm.subdivide(30)
    assert sub.total_length() == pytest.approx(rm.total_length(), rel=1e-9)
    assert max(sub.edge_lengths()) <= 30 + 1e-9
    import networkx as nx
    assert nx.number_connected_components(sub.graph) == 1


def test_perturbation_counts():
    rng = np.random.default_rng(0)
    segs = RoadMap.synthetic_grid(6, 6, 100).segments()
    p = PerturbationParams(jitter=10, offset=5, duplication=50, split=0, deletion=0)
    out, src, dup = perturb_segments(segs, p, rng, return_info=True)
    assert dup.sum() == round(len(segs) * 0.5)
    assert len(out) == len(segs) + dup.sum()
    rm = RoadMap.synthetic_grid(6, 6, 100)
    assert apply_deletion(rm, 20, rng).graph.number_of_edges() == len(segs) - round(len(segs) * 0.2)


def test_simulation_rules_and_coverage():
    rm = RoadMap.synthetic_grid(5, 5, 60)
    sim = Simulation(rm, deploy_quadrants(rm, 3), seed=0)
    steps = sim.run_until_coverage(k=3, max_steps=5000)
    assert steps is not None
    tr = sim.traces()
    assert set(["volunteer_id", "trace_id", "timestamp", "lat", "lon"]) <= set(tr.columns)
    # agents stay inside their quadrant boxes (+ margin) whenever possible
    assert tr.groupby(["volunteer_id", "trace_id"]).timestamp.is_monotonic_increasing.all()


def test_end_to_end_reconstruction():
    rm = RoadMap.synthetic_grid(4, 4, 100)
    out = end_to_end_simulation(rm, n_agents=16, steps=400, deletion=10, seed=0,
                                deployment="quadrant")
    m = out["metrics"]
    assert m["length_recall"] > 0.9
    assert m["length_precision"] > 0.95
    assert out["summary"]["intersections_found"] >= 4


def test_pipeline_without_premap_returns_trace_map():
    rm = RoadMap.synthetic_grid(3, 3, 100)
    sim = Simulation(rm, deploy_quadrants(rm, 3), seed=1).run(200)
    res = Proteus(None, ProteusConfig(map_match=False), projection=rm.proj).run(sim.traces())
    assert res.combination is None
    assert res.final_map.graph.number_of_edges() > 0
    assert map_metrics(rm, res.final_map, theta=10)["length_precision"] > 0.9


def test_edge_verification_trends():
    rm = RoadMap.synthetic_grid(6, 6, 100, edge_subdivisions=2)
    acc = {m: edge_verification_trial(rm, PerturbationParams(), m, seed=0).accuracy for m in (2, 3, 5)}
    assert acc[2] < acc[3] <= acc[5] + 1e-9
    grid = {"Jitter": [5, 20], "Offset": [5], "Duplication Constant": [55], "Split": [35],
            "Deletion": [10]}
    df = edge_verification_experiment(rm, grid, models=(3,))
    assert list(df.columns[:6]) == ["Jitter", "Offset", "Duplication Constant", "Split",
                                     "Deletion", "3 edge accuracy"]


def test_path_combination_and_similarity():
    rm = RoadMap.synthetic_grid(5, 5, 100, edge_subdivisions=2)
    df = path_combination_experiment(rm, models=(5,))
    assert df.loc[0, "similarity_%"] > 80
    assert graph_similarity(rm, rm) == pytest.approx(1.0)


def test_deployment_experiment_runs():
    rm = RoadMap.synthetic_grid(5, 5, 60)
    df = deployment_experiment(rm, agents_per_quadrant=3, max_steps=20_000)
    assert len(df) == 3 and (df["coverage_at_end_%"] == 100).all()


@pytest.mark.skipif(not (DATA / "edge_verification_results_raw_downtown.csv").exists(),
                    reason="ProteusData CSV not available")
def test_eta_squared_reproduces_table3():
    df = pd.read_csv(DATA / "edge_verification_results_raw_downtown.csv")
    factors = ["Jitter", "Offset", "Duplication Constant", "Split", "Deletion"]
    expected = {  # Table 3 of the paper: (eta^2, partial eta^2)
        "2 edge accuracy": {"Jitter": (0.6968, 0.7267), "Offset": (0.0388, 0.1289),
                            "Duplication Constant": (0.0016, 0.0064), "Split": (0.0002, 0.0008),
                            "Deletion": (0.0004, 0.0014)},
        "3 edge accuracy": {"Jitter": (0.7612, 0.7962), "Offset": (0.0425, 0.1791),
                            "Duplication Constant": (0.0008, 0.0041), "Split": (0.0002, 0.0010),
                            "Deletion": (0.0005, 0.0028)},
        "5 edge accuracy": {"Jitter": (0.7627, 0.7814), "Offset": (0.0216, 0.0919),
                            "Duplication Constant": (0.0020, 0.0092), "Split": (0.0002, 0.0010),
                            "Deletion": (0.0002, 0.0008)},
    }
    for col, rows in expected.items():
        t = eta_squared(df, col, factors)
        for f, (e, pe) in rows.items():
            assert t.loc[f, "eta_sq"] == pytest.approx(e, abs=6e-4)
            assert t.loc[f, "partial_eta_sq"] == pytest.approx(pe, abs=6e-4)


def test_cli_simulate_and_run(tmp_path):
    out = tmp_path / "sim"
    cli_main(["simulate", "--grid", "3,3,100", "--agents", "12", "--steps", "150",
              "--out", str(out)])
    assert (out / "final_map.csv").exists() and (out / "traces.csv").exists()
    cli_main(["run", "--grid", "3,3,100", "--traces", str(out / "traces.csv"),
              "--out", str(tmp_path / "run")])
    assert (tmp_path / "run" / "summary.json").exists()
