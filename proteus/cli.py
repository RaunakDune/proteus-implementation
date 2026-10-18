"""Command line interface: ``python -m proteus <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .roadmap import RoadMap


# --------------------------------------------------------------------------- #
def _add_map_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("road map (pick one)")
    g.add_argument("--map", help="edge list CSV with columns x0,y0,x1,y1 (lon/lat)")
    g.add_argument("--osm", help="'lat,lon,dist_m,network_type' -- download with OSMnx")
    g.add_argument("--grid", help="'rows,cols,spacing_m' -- synthetic street grid")
    g.add_argument("--campus", help="'size_m,n_paths' -- synthetic curvy campus network")


def _load_map(a) -> RoadMap:
    if a.map:
        return RoadMap.from_edge_csv(a.map)
    if a.osm:
        lat, lon, dist, kind = a.osm.split(",")
        return RoadMap.from_osmnx(float(lat), float(lon), float(dist), kind.strip())
    if a.grid:
        r, c, s = a.grid.split(",")
        return RoadMap.synthetic_grid(int(r), int(c), float(s), edge_subdivisions=2)
    if a.campus:
        size, n = a.campus.split(",")
        return RoadMap.synthetic_campus(float(size), int(n), seed=getattr(a, "seed", 0))
    raise SystemExit("one of --map / --osm / --grid / --campus is required")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Proteus parameters")
    g.add_argument("--threshold", type=float, default=10.0, help="theta, max GPS noise (m)")
    g.add_argument("--min-support", type=int, default=3, help="n_p needed to verify a point")
    g.add_argument("--count-mode", default="volunteers",
                   help="points | traces | volunteers | <column name>")
    g.add_argument("--no-map-match", action="store_true")
    g.add_argument("--angle", type=float, default=15.0, help="omega, tolerance angle (deg)")
    g.add_argument("--segment-length", type=float, default=20.0, help="l (m)")
    g.add_argument("--intersection-radius", type=float, default=15.0)
    g.add_argument("--min-agreement", type=int, default=2)
    g.add_argument("--agreement-reference", default="verified", choices=["verified", "map"])


def _config(a):
    from .pipeline import ProteusConfig

    return ProteusConfig(threshold=a.threshold, min_support=a.min_support,
                         count_mode=a.count_mode, map_match=not a.no_map_match,
                         angle_tolerance=a.angle, segment_length=a.segment_length,
                         intersection_radius=a.intersection_radius,
                         min_agreement=a.min_agreement,
                         agreement_reference=a.agreement_reference)


def _write_result(result, out: Path, pre_map=None, traces=None, truth=None) -> None:
    from .verification import verified_edges_to_frame

    out.mkdir(parents=True, exist_ok=True)
    result.final_map.to_edge_csv(out / "final_map.csv")
    result.trace_map.to_edge_csv(out / "trace_map.csv")
    verified_edges_to_frame(result.verification.edges, result.final_map.proj).to_csv(
        out / "verified_edges.csv", index=False)
    if result.intersections.intersections:
        p = np.array(result.intersections.intersections)
        lon, lat = result.final_map.proj.to_lonlat(p[:, 0], p[:, 1])
        pd.DataFrame({"lon": lon, "lat": lat}).to_csv(out / "intersections.csv", index=False)
    try:
        from .plotting import plot_result

        plot_result(result, pre_map=pre_map, traces=traces, path=out / "map.png")
    except Exception as e:  # pragma: no cover - plotting is optional
        print(f"(plot skipped: {e})", file=sys.stderr)


# --------------------------------------------------------------------------- #
def cmd_run(a):
    from .pipeline import Proteus
    from .traces import load_traces, with_xy

    pre = _load_map(a) if (a.map or a.osm or a.grid or a.campus) else None
    traces = load_traces(a.traces)
    result = Proteus(pre, _config(a)).run(traces)
    out = Path(a.out)
    proj = result.final_map.proj
    _write_result(result, out, pre_map=pre, traces=with_xy(traces, proj))
    (out / "summary.json").write_text(json.dumps(result.summary(), indent=2, default=str))
    print(json.dumps(result.summary(), indent=2, default=str))


def cmd_simulate(a):
    from .experiments import end_to_end_simulation
    from .simulation import GPSModel
    from .traces import save_traces

    pre = _load_map(a)
    res = end_to_end_simulation(pre, n_agents=a.agents, steps=a.steps, deletion=a.deletion,
                                gps=GPSModel(sigma=a.gps_sigma), config=_config(a),
                                deployment=a.deployment, seed=a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_traces(res["traces"], out / "traces.csv")
    res["post_disaster"].to_edge_csv(out / "post_disaster_truth.csv")
    _write_result(res["result"], out, pre_map=pre, traces=res["traces"])
    report = {"summary": res["summary"], "metrics": res["metrics"]}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


def cmd_assign(a):
    from .assignment import distribute_agents

    rm = _load_map(a)
    starts = distribute_agents(rm, a.agents, a.area)
    rows = []
    for s in starts:
        lon, lat = rm.proj.to_lonlat(*s.xy)
        nd = rm.graph.nodes[s.node]
        rows.append({"agent": s.agent_id, "lon": float(lon), "lat": float(lat),
                     "start_node_lon": nd["x"], "start_node_lat": nd["y"]})
    df = pd.DataFrame(rows)
    if a.out:
        df.to_csv(a.out, index=False)
    print(df.to_string(index=False))


def cmd_edge_experiment(a):
    from .experiments import (PAPER_GRID, QUICK_GRID, edge_verification_experiment,
                              edge_verification_summary)

    rm = _load_map(a)
    print("Table 2 style summary (paper default parameters):")
    print(edge_verification_summary(rm, seed=a.seed, threshold=a.threshold).to_string(index=False))
    if a.summary_only:
        return
    grid = QUICK_GRID if a.quick else PAPER_GRID
    df = edge_verification_experiment(rm, grid, repeats=a.repeats, seed=a.seed, jobs=a.jobs,
                                      progress=True, threshold=a.threshold)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}")


def cmd_combination_experiment(a):
    from .experiments import path_combination_experiment

    rm = _load_map(a)
    print(path_combination_experiment(rm, config=_config(a), seed=a.seed,
                                      similarity_method=a.similarity).to_string(index=False))


def cmd_deployment_experiment(a):
    from .experiments import deployment_experiment

    rm = _load_map(a)
    print(deployment_experiment(rm, agents_per_quadrant=a.agents_per_quadrant, k=a.k,
                                max_steps=a.max_steps, seed=a.seed).to_string(index=False))


def cmd_anova(a):
    from .evaluation import eta_squared, fit_anova_model

    df = pd.read_csv(a.csv)
    factors = [f for f in ["Jitter", "Offset", "Duplication Constant", "Split", "Deletion"]
               if f in df.columns]
    models = [m for m in (2, 3, 5) if f"{m} edge accuracy" in df.columns]
    rows = []
    for m in models:
        t = eta_squared(df, f"{m} edge accuracy", factors)
        for f in factors:
            rows.append({"model": f"{m}-edge", "factor": f, "eta_sq": round(t.loc[f, "eta_sq"], 4),
                         "partial_eta_sq": round(t.loc[f, "partial_eta_sq"], 4),
                         "p_value": t.loc[f, "PR(>F)"]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if a.plots:
        from .plotting import anova_diagnostics, factor_comparison, jitter_boxplots

        out = Path(a.plots)
        out.mkdir(parents=True, exist_ok=True)
        for m in models:
            anova_diagnostics(fit_anova_model(df, f"{m} edge accuracy", factors),
                              path=out / f"x{m}_anova_diagnostics.png", title=f"{m}-edge ANOVA")
        jitter_boxplots(df, models, path=out / "jitter_significance.png")
        for f in factors[1:]:
            factor_comparison(df, f, model=max(models), path=out / f"x{max(models)}_jitter_vs_{f.replace(' ', '_')}.png")
        print(f"plots written to {out}")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="proteus", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="run Proteus on a trace CSV")
    _add_map_args(s)
    _add_config_args(s)
    s.add_argument("--traces", required=True, help="CSV: volunteer_id,trace_id,timestamp,lat,lon")
    s.add_argument("--out", default="proteus_out")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("simulate", help="agent-based simulation + Proteus + scoring")
    _add_map_args(s)
    _add_config_args(s)
    s.add_argument("--agents", type=int, default=24)
    s.add_argument("--steps", type=int, default=1000)
    s.add_argument("--deletion", type=float, default=10.0, help="%% of roads destroyed")
    s.add_argument("--gps-sigma", type=float, default=3.0)
    s.add_argument("--deployment", default="quadrant",
                   choices=["algorithm1", "quadrant", "poisson", "random"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="proteus_sim")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("assign", help="Algorithm 1: starting locations for agents")
    _add_map_args(s)
    s.add_argument("--agents", type=int, required=True)
    s.add_argument("--area", type=float, required=True, help="max area per agent (m^2)")
    s.add_argument("--out")
    s.set_defaults(func=cmd_assign)

    s = sub.add_parser("edge-experiment", help="Sec 5.3/5.4 edge verification grid")
    _add_map_args(s)
    s.add_argument("--threshold", type=float, default=10.0)
    s.add_argument("--quick", action="store_true", help="smaller grid (144 rows)")
    s.add_argument("--summary-only", action="store_true")
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="edge_verification_results.csv")
    s.set_defaults(func=cmd_edge_experiment)

    s = sub.add_parser("combination-experiment", help="Sec 5.5 path combination test")
    _add_map_args(s)
    _add_config_args(s)
    s.add_argument("--similarity", default="anchored", choices=["anchored", "networkx"])
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_combination_experiment)

    s = sub.add_parser("deployment-experiment", help="Sec 5.6 volunteer deployment test")
    _add_map_args(s)
    s.add_argument("--agents-per-quadrant", type=int, default=4)
    s.add_argument("--k", type=int, default=3)
    s.add_argument("--max-steps", type=int, default=100_000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_deployment_experiment)

    s = sub.add_parser("anova", help="eta-squared ANOVA on an edge-experiment CSV (Table 3)")
    s.add_argument("csv")
    s.add_argument("--plots", help="directory for diagnostic plots")
    s.set_defaults(func=cmd_anova)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
