"""Experiments from Sec 5 of the paper.

* :func:`edge_verification_trial` / :func:`edge_verification_experiment`
  -- Sec 5.3 (Table 2) and the parameter study of Sec 5.4 (Table 3, Figs 11-15).
  The output of the full grid has the same columns as
  ``edge_verification_results_raw_*.csv`` in the ProteusData repository.
* :func:`path_combination_experiment` -- Sec 5.5 (Table 4).
* :func:`deployment_experiment` -- Sec 5.6 (Table 5).
* :func:`end_to_end_simulation` -- the whole system on agent-based traces
  (Figs 7-8).

In this simulator the "x-edge model" (x = 2, 3, 5) is *x independent volunteer
collections* of every edge (Sec 5.1; Fig. 7, "using 2 agents for
verification").  Points are verified with Algorithm 2 (``n_p >= 3``).
"""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assignment import deploy_poisson, deploy_quadrants, deploy_random, distribute_agents
from .combination import combine_maps
from .evaluation import edge_reproduction, graph_similarity, map_metrics
from .geo import SegmentIndex
from .intersections import (build_combined_map, graph_to_segments, handle_intersections,
                            verified_edges_to_segments)
from .perturbation import (PerturbationParams, apply_deletion, perturb_segments,
                           segments_to_traces)
from .pipeline import Proteus, ProteusConfig
from .roadmap import RoadMap
from .simulation import AgentRules, GPSModel, Simulation
from .verification import verify_traces

PAPER_GRID = {
    "Jitter": [5, 10, 15, 20],
    "Offset": [5, 8, 11, 14, 17],
    "Duplication Constant": [30, 35, 40, 45, 50, 55, 60, 65, 70],
    "Split": [10, 20, 30, 40],
    "Deletion": [0, 10, 20],
}
QUICK_GRID = {
    "Jitter": [5, 10, 15, 20],
    "Offset": [5, 11, 17],
    "Duplication Constant": [30, 50, 70],
    "Split": [10, 40],
    "Deletion": [0, 20],
}
MODELS = (5, 3, 2)


# --------------------------------------------------------------------------- #
# Edge verification (Sec 5.3)
# --------------------------------------------------------------------------- #
@dataclass
class EdgeTrialResult:
    model: int
    input_edges: int
    verified_edges: int
    accuracy: float           # % of ground-truth edges reproduced
    precision: float          # % of verified geometry lying on ground truth
    seconds: float
    edges: list | None = None  # VerifiedEdge objects (kept for path combination)


def simulate_volunteer_copies(truth: RoadMap, params: PerturbationParams, n_volunteers: int,
                              rng: np.random.Generator, spacing: float = 3.0) -> pd.DataFrame:
    """Traces from ``n_volunteers`` independent noisy copies of ``truth``.

    Every polyline becomes its own trace (so a verified edge never bridges two
    unrelated edges).  The ``collection`` column groups what was collected
    together: a volunteer's walk over the map is one collection (adjacent
    edges of the same walk cannot verify each other), while every duplicated
    edge is an extra, independent re-collection of that edge.
    """
    segs = truth.segments()
    rows = []
    for v in range(n_volunteers):
        polys, src, dup = perturb_segments(segs, params, rng, return_info=True)
        coll = [f"{v}/dup{k}" if d else f"{v}" for k, d in enumerate(dup)]
        rows += segments_to_traces(polys, v, spacing, params.gps_sigma, rng, collections=coll)
    df = pd.DataFrame(rows, columns=["volunteer_id", "trace_id", "timestamp", "x", "y", "collection"])
    lon, lat = truth.proj.to_lonlat(df.x.to_numpy(), df.y.to_numpy())
    df["lon"], df["lat"] = lon, lat
    return df


def edge_verification_trial(truth: RoadMap, params: PerturbationParams, model: int, *,
                            threshold: float = 10.0, min_support: int = 3,
                            count_mode: str = "collection", map_match: bool = False,
                            spacing: float = 3.0, seed: int = 0,
                            post_disaster: RoadMap | None = None,
                            keep_edges: bool = False) -> EdgeTrialResult:
    """One run of the edge verification test for a single x-edge model."""
    rng = np.random.default_rng(seed)
    post = post_disaster if post_disaster is not None else apply_deletion(truth, params.deletion, rng)
    t0 = time.perf_counter()
    traces = simulate_volunteer_copies(post, params, model, rng, spacing)
    res = verify_traces(traces, threshold, road_map=truth, min_support=min_support,
                        count_mode=count_mode, map_match=map_match)
    secs = time.perf_counter() - t0
    pts = np.vstack([e.points for e in res.edges]) if res.edges else np.zeros((0, 2))
    truth_segs = post.segments()
    rep = edge_reproduction(truth_segs, points=pts, theta=threshold)
    if len(pts):   # share of verified points lying within theta of the true map
        _, d = SegmentIndex(truth_segs, sample_spacing=2.0).nearest(pts, threshold)
        prec = float(np.mean(np.isfinite(d)) * 100)
    else:
        prec = 0.0
    return EdgeTrialResult(model=model, input_edges=len(truth_segs), verified_edges=len(res.edges),
                           accuracy=float(rep.mean() * 100) if len(rep) else 0.0,
                           precision=prec, seconds=secs,
                           edges=res.edges if keep_edges else None)


def _grid_rows(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _run_grid_row(args):
    truth, row, models, kw, seed = args
    params = PerturbationParams(jitter=row["Jitter"], offset=row["Offset"],
                                duplication=row["Duplication Constant"], split=row["Split"],
                                deletion=row["Deletion"])
    rng = np.random.default_rng(seed)
    post = apply_deletion(truth, params.deletion, rng)
    out = dict(row)
    for m in models:
        r = edge_verification_trial(truth, params, m, seed=seed + m, post_disaster=post, **kw)
        out[f"{m} edge accuracy"] = r.accuracy
        out[f"{m} edge precision"] = r.precision
        out[f"{m} edge verified"] = r.verified_edges
        out[f"{m} edge seconds"] = r.seconds
    return out


def edge_verification_experiment(truth: RoadMap, grid: dict | None = None,
                                 models=MODELS, *, repeats: int = 1, seed: int = 0,
                                 jobs: int = 1, progress: bool = False, **trial_kw) -> pd.DataFrame:
    """Run the parameter grid of Sec 5.4.

    ``grid`` defaults to the paper's 4 x 5 x 9 x 4 x 3 = 2160 combinations
    (:data:`PAPER_GRID`).  The first eight columns of the result match the
    ProteusData CSV files; extra columns hold precision, verified-edge counts
    and run times.
    """
    grid = grid or PAPER_GRID
    tasks = []
    for r in range(repeats):
        for i, row in enumerate(_grid_rows(grid)):
            tasks.append((truth, row, tuple(models), trial_kw, seed + 1000 * r + 7 * i))
    rows = []
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            for k, out in enumerate(ex.map(_run_grid_row, tasks, chunksize=4)):
                rows.append(out)
                if progress and k % 50 == 0:
                    print(f"  {k}/{len(tasks)}", flush=True)
    else:
        for k, t in enumerate(tasks):
            rows.append(_run_grid_row(t))
            if progress and k % 50 == 0:
                print(f"  {k}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows)
    lead = list(grid) + [f"{m} edge accuracy" for m in models]
    return df[lead + [c for c in df.columns if c not in lead]]


def edge_verification_summary(truth: RoadMap, params: PerturbationParams | None = None,
                              models=(2, 3, 5), seed: int = 0, **trial_kw) -> pd.DataFrame:
    """Table 2: input edges, verified edges, accuracy and time per model."""
    params = params or PerturbationParams()
    rng = np.random.default_rng(seed)
    post = apply_deletion(truth, params.deletion, rng)
    rows = []
    for m in models:
        r = edge_verification_trial(truth, params, m, seed=seed + m, post_disaster=post, **trial_kw)
        rows.append({"model": f"{m}-edge", "input_edges": r.input_edges,
                     "verified_edges": r.verified_edges, "accuracy_%": round(r.accuracy, 2),
                     "precision_%": round(r.precision, 2), "time_s": round(r.seconds, 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Path combination (Sec 5.5)
# --------------------------------------------------------------------------- #
def path_combination_experiment(truth: RoadMap, params: PerturbationParams | None = None,
                                models=(2, 3, 5), *, config: ProteusConfig | None = None,
                                seed: int = 0, similarity_method: str = "anchored",
                                **trial_kw) -> pd.DataFrame:
    """Table 4: combine the verified edges of each model into one map
    (Algorithms 3-6) and compare it with the ground truth."""
    params = params or PerturbationParams()
    cfg = config or ProteusConfig()
    rng = np.random.default_rng(seed)
    post = apply_deletion(truth, params.deletion, rng)
    rows = []
    for m in models:
        r = edge_verification_trial(truth, params, m, seed=seed + m, post_disaster=post,
                                    threshold=cfg.threshold, keep_edges=True, **trial_kw)
        g = build_combined_map(r.edges, spacing=cfg.graph_spacing,
                               merge_radius=cfg.merge_radius or cfg.threshold,
                               heading_tolerance=cfg.heading_tolerance,
                               smooth_window=cfg.smooth_window)
        inter = handle_intersections(g, cfg.intersection_radius)
        if cfg.segment_source == "verified":
            segs = verified_edges_to_segments(r.edges, cfg.segment_length, spacing=cfg.graph_spacing,
                                              smooth_window=cfg.smooth_window)
        else:
            segs = graph_to_segments(inter.graph, cfg.segment_length)
        comb = combine_maps(truth, segs, cfg.threshold, max(cfg.angle_tolerance, params.offset),
                            subdivision_length=cfg.segment_length,
                            min_subdivision_length=cfg.min_subdivision_length,
                            min_agreement=cfg.min_agreement)
        truth_sub = post.subdivide(comb.subdivision_length)
        sim = graph_similarity(truth_sub, comb.map, tolerance=1.0, method=similarity_method)
        mm = map_metrics(post, comb.map, theta=cfg.threshold)
        rows.append({"model": f"{m}-edge", "input_edges": r.verified_edges,
                     "similarity_%": round(sim * 100, 2),
                     "length_recall_%": round(mm["length_recall"] * 100, 2),
                     "length_precision_%": round(mm["length_precision"] * 100, 2),
                     "intersections": len(inter.intersections)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Volunteer deployment (Sec 5.6)
# --------------------------------------------------------------------------- #
def deployment_experiment(road_map: RoadMap, agents_per_quadrant: int = 4, n_split: int = 2,
                          k: int = 3, max_steps: int = 100_000, seed: int = 0,
                          rules: AgentRules | None = None) -> pd.DataFrame:
    """Table 5: map coverage time for the three deployment approaches.

    The same total number of agents is used by every approach.
    """
    rm = road_map.largest_component()
    n_agents = agents_per_quadrant * n_split * n_split
    nodes = list(rm.graph.nodes)
    rows = []
    for name, starts, policy in (
        ("Quadrant-based", deploy_quadrants(rm, agents_per_quadrant, n_split), "rules"),
        ("Whole map", deploy_poisson(rm, n_agents, seed=seed), "rules"),
        ("Random deployment", deploy_random(rm, n_agents, seed=seed), "random"),
    ):
        sim = Simulation(rm, starts, policy=policy, rules=rules, seed=seed)
        steps = sim.run_until_coverage(k=k, max_steps=max_steps, nodes=nodes)
        rows.append({"approach": name, "agents": n_agents,
                     "coverage_steps": steps if steps is not None else f">{max_steps}",
                     "coverage_at_end_%": round(sim.coverage(k, nodes) * 100, 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# End to end (agent based traces -> Proteus)
# --------------------------------------------------------------------------- #
def end_to_end_simulation(pre_map: RoadMap, *, n_agents: int = 12, steps: int = 600,
                          deletion: float = 10.0, gps: GPSModel | None = None,
                          rules: AgentRules | None = None, config: ProteusConfig | None = None,
                          deployment: str = "algorithm1", redundancy: float = 3.0,
                          seed: int = 0) -> dict:
    """Simulate a disaster, let volunteers collect traces on the post-disaster
    network, run Proteus and score the reconstruction.

    Returns a dict with the result object, the post-disaster truth, the
    traces and metrics (recall/precision vs. the post-disaster map and the
    fraction of destroyed road length that Proteus wrongly kept).
    """
    rng = np.random.default_rng(seed)
    pre = pre_map.largest_component()
    post = apply_deletion(pre, deletion, rng)
    if deployment == "algorithm1":
        # each agent may cover `redundancy` times its fair share of the AOI, so
        # every cell is shared by ~redundancy volunteers (needed for n_p >= 3)
        area = pre.area_m2()
        starts = distribute_agents(post, n_agents, redundancy * area / n_agents)
    elif deployment == "quadrant":
        starts = deploy_quadrants(post, max(1, n_agents // 4))
    elif deployment == "poisson":
        starts = deploy_poisson(post, n_agents, seed=seed)
    else:
        starts = deploy_random(post, n_agents, seed=seed)
    sim = Simulation(post, starts, rules=rules, gps=gps, seed=seed).run(steps)
    traces = sim.traces()
    cfg = config or ProteusConfig()
    result = Proteus(pre, cfg).run(traces)
    sub_len = result.combination.subdivision_length if result.combination else cfg.segment_length
    metrics = map_metrics(post, result.final_map, theta=cfg.threshold)
    deleted = pre.with_edges([e for e in pre.edge_list() if not post.graph.has_edge(*e)])
    if deleted.graph.number_of_edges():
        # how much of the destroyed road network is (wrongly) still in the map
        dm = map_metrics(deleted, result.final_map, theta=1.0)
        metrics["destroyed_length_kept_%"] = round(dm["length_recall"] * 100, 2)
        metrics["destroyed_edges_kept_%"] = round(dm["edge_recall"] * 100, 2)
    metrics["coverage_k3_%"] = round(sim.coverage(3) * 100, 2)
    metrics["similarity_%"] = round(100 * graph_similarity(post.subdivide(sub_len),
                                                           result.final_map, tolerance=1.0), 2)
    return {"result": result, "post_disaster": post, "traces": traces,
            "simulation": sim, "metrics": metrics, "summary": result.summary()}


__all__ = [
    "PAPER_GRID", "QUICK_GRID", "MODELS", "EdgeTrialResult", "simulate_volunteer_copies",
    "edge_verification_trial", "edge_verification_experiment", "edge_verification_summary",
    "path_combination_experiment", "deployment_experiment", "end_to_end_simulation",
]
