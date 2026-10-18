"""Re-run the experiments of Sec 5 and write tables/figures to ``results/``.

    python examples/reproduce_paper.py [--osm] [--full-grid] [--jobs N]

By default the two maps of the paper are approximated by the edge lists
shipped in ``data/`` (``houston_downtown_map_gen.csv`` and
``university_map_gen.csv`` from the ProteusData repository).  With ``--osm``
the original OpenStreetMap networks are downloaded with OSMnx instead
(Downtown Houston: drive network, UH campus: walk network).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from proteus import ProteusConfig, RoadMap, eta_squared
from proteus.experiments import (PAPER_GRID, QUICK_GRID, deployment_experiment,
                                 edge_verification_experiment, edge_verification_summary,
                                 end_to_end_simulation, path_combination_experiment)
from proteus.plotting import jitter_boxplots, plot_result

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
FACTORS = ["Jitter", "Offset", "Duplication Constant", "Split", "Deletion"]


def load_maps(osm: bool) -> dict[str, RoadMap]:
    if osm:
        return {"Downtown Houston": RoadMap.from_osmnx(29.755733, -95.364688, 800, "drive"),
                "University of Houston": RoadMap.from_osmnx(29.721489, -95.343374, 600, "walk")}
    return {"Downtown Houston": RoadMap.from_edge_csv(DATA / "houston_downtown_map_gen.csv"),
            "University of Houston": RoadMap.from_edge_csv(DATA / "university_map_gen.csv")}


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", action="store_true")
    ap.add_argument("--full-grid", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    maps = load_maps(a.osm)
    report = ["# Reproduction run\n"]

    # Table 3 from the published raw data ------------------------------------
    for name, f in (("Downtown Houston", "edge_verification_results_raw_downtown.csv"),
                    ("University of Houston", "edge_verification_results_raw_UH.csv")):
        if not (DATA / f).exists():
            continue
        df = pd.read_csv(DATA / f)
        rows = []
        for m in (2, 3, 5):
            t = eta_squared(df, f"{m} edge accuracy", FACTORS)
            for fac in FACTORS:
                rows.append({"model": f"{m}-edge", "factor": fac,
                             "eta_sq": round(t.loc[fac, "eta_sq"], 4),
                             "partial_eta_sq": round(t.loc[fac, "partial_eta_sq"], 4)})
        report += [f"## Table 3 from published ProteusData results -- {name}\n", md(pd.DataFrame(rows)), ""]

    # Table 2 / Table 4 / Table 5 --------------------------------------------
    for name, rm in maps.items():
        report += [f"## {name}  ({rm})\n"]
        t2 = edge_verification_summary(rm, seed=0)
        report += ["### Edge verification (Table 2)\n", md(t2), ""]
        t4 = path_combination_experiment(rm, seed=0)
        report += ["### Path combination (Table 4)\n", md(t4), ""]
        t5 = deployment_experiment(rm, agents_per_quadrant=4, max_steps=60_000, seed=0)
        report += ["### Volunteer deployment, 3-agent coverage (Table 5)\n", md(t5), ""]
        grid = PAPER_GRID if a.full_grid else QUICK_GRID
        df = edge_verification_experiment(rm, grid, jobs=a.jobs, seed=0)
        tag = name.split()[0].lower()
        df.to_csv(OUT / f"edge_verification_sim_{tag}.csv", index=False)
        jitter_boxplots(df, path=OUT / f"jitter_{tag}.png")
        rows = []
        for m in (2, 3, 5):
            t = eta_squared(df, f"{m} edge accuracy", FACTORS)
            for fac in FACTORS:
                rows.append({"model": f"{m}-edge", "factor": fac,
                             "eta_sq": round(t.loc[fac, "eta_sq"], 4),
                             "partial_eta_sq": round(t.loc[fac, "partial_eta_sq"], 4)})
        report += ["### Parameter study on simulated data (Table 3 style)\n", md(pd.DataFrame(rows)), ""]

    # End-to-end agent based runs (Figs 7-8 style) ---------------------------
    for name, rm in (("synthetic grid", RoadMap.synthetic_grid(8, 8, 100, edge_subdivisions=2)),
                     ("synthetic campus", RoadMap.synthetic_campus(800, 50, seed=2))):
        out = end_to_end_simulation(rm, n_agents=32, steps=1500, deletion=10, seed=0,
                                    deployment="quadrant", config=ProteusConfig())
        tag = name.split()[1]
        plot_result(out["result"], pre_map=rm, traces=out["traces"], path=OUT / f"end_to_end_{tag}.png",
                    title=f"Proteus on a {name} (red: reconstructed post-disaster map)")
        report += [f"## End-to-end simulation: {name}\n", "```json",
                   json.dumps({"summary": out["summary"], "metrics": out["metrics"]}, indent=2), "```", ""]

    (OUT / "REPORT.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
