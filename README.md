# Proteus: combining paths collected using crowdsensing

The reference implementation of Proteus, from the paper

> R. Sarbajna, C. F. Eick, J. Ni. **Proteus: A system for combining paths collected using
> crowdsensing.** *GeoInformatica* 30:7 (2026). https://doi.org/10.1007/s10707-026-00567-1

After a disaster, volunteers walk or drive around the affected area and record GPS
traces. Proteus checks those traces against each other and against the pre-disaster map,
then builds a map of the roads that are still passable. The paper evaluated Proteus in
simulation only. This repository turns Algorithms 1–6 and the volunteer movement rules into
a pipeline that runs on real GPS traces. It also includes an agent-based simulator, the
experiments of Section 5, and the ANOVA analysis, which reproduces Table 3 exactly from the
published [ProteusData](https://github.com/UH-DAIS/ProteusData) results.

```
GPS traces ──► Alg 2  point-wise verification ──► verified edges
                                                     │
            combined map M ◄──────────────────────────┘
                 │
                 ▼
          Alg 3/4  intersection detection + handling ──► corrected map M_C (trace-only map)
                 │
pre-disaster ──► Alg 5/6  agreement + map combination ──► post-disaster map
map M
```

## Install

```bash
pip install -e .                 # numpy, scipy, pandas, networkx
pip install -e ".[analysis]"     # + statsmodels, matplotlib (ANOVA, figures)
pip install -e ".[osm]"          # + osmnx (download OpenStreetMap road networks)
pip install -e ".[dev]"          # + pytest
```

## Quick start

```python
import numpy as np
from proteus import (Proteus, ProteusConfig, RoadMap, Simulation, GPSModel,
                     apply_deletion, deploy_quadrants, map_metrics)

pre_map  = RoadMap.from_osmnx(29.755733, -95.364688, dist_m=800, network_type="drive")
# or: RoadMap.from_edge_csv("my_map.csv")  /  RoadMap.synthetic_grid(6, 6, 100)

# real data: a CSV with volunteer_id, trace_id, timestamp, lat, lon
# traces = proteus.traces.load_traces("traces.csv")
# ... or simulate volunteers on a damaged copy of the map:
post_map = apply_deletion(pre_map, 10, np.random.default_rng(0))
sim = Simulation(post_map, deploy_quadrants(post_map, 6), gps=GPSModel(sigma=3)).run(800)
traces = sim.traces()

result = Proteus(pre_map, ProteusConfig(threshold=10)).run(traces)
print(result.summary())
print(map_metrics(post_map, result.final_map))
result.final_map.to_edge_csv("post_disaster_map.csv")
```

Command line (`python -m proteus ...` or `proteus ...`):

```bash
proteus run       --map premap.csv --traces traces.csv --out out/      # Proteus on real traces
proteus simulate  --grid 6,6,100 --agents 24 --steps 1000 --out sim/   # ABM + Proteus + scores
proteus assign    --osm 29.7557,-95.3647,800,drive --agents 20 --area 40000   # Algorithm 1
proteus edge-experiment        --map data/houston_downtown_map_gen.csv --quick --jobs 4
proteus combination-experiment --map data/houston_downtown_map_gen.csv
proteus deployment-experiment  --grid 10,10,80
proteus anova data/edge_verification_results_raw_downtown.csv --plots figs/   # Table 3, Figs 9-15
```

Every map option accepts `--map` (edge list CSV `x0,y0,x1,y1` in lon/lat), `--osm
lat,lon,dist,network_type`, `--grid rows,cols,spacing` or `--campus size,n_paths`.

## Paper → code

| Paper | Code |
|---|---|
| Sec 3 traces `<VolunteerID, location, timestamp>` | `proteus/traces.py` |
| Sec 4.1 movement rules 1–4 | `proteus/simulation.py` (`AgentRules`, `Simulation`) |
| **Algorithm 1**: agent distribution in the AOI | `assignment.distribute_agents` |
| **Algorithm 2**: trace verification | `verification.verify_traces`, `verification.count_support` |
| Combined map from verified edges | `intersections.build_combined_map` |
| **Algorithm 3**: intersection detection `INTERSECT(A, E)` | `intersections.intersect` |
| **Algorithm 4**: intersection handler | `intersections.handle_intersections` |
| **Algorithm 5**: agreement function `AGREE(A, B, θ, ω)` | `combination.agree` (vectorised: `combination.agreement_matrix`) |
| **Algorithm 6**: map combination | `combination.combine_maps` |
| Whole pipeline (Fig 1) | `pipeline.Proteus`, `pipeline.ProteusConfig` |
| Sec 5.3 simulation parameters (Table 1) | `perturbation.PerturbationParams`, `perturb_segments`, `apply_deletion` |
| Sec 5.3 edge verification test (Table 2) | `experiments.edge_verification_trial / _summary / _experiment` |
| Sec 5.2/5.4 ANOVA, η² (Table 3, Figs 9–15) | `evaluation.eta_squared`, `plotting.anova_diagnostics`, `plotting.jitter_boxplots` |
| Sec 5.5 path combination test, GED similarity (Table 4) | `experiments.path_combination_experiment`, `evaluation.graph_similarity` |
| Sec 5.6 deployment strategies and coverage time (Table 5) | `assignment.deploy_quadrants / deploy_poisson / deploy_random`, `experiments.deployment_experiment` |
| Figs 7–8 generated maps | `experiments.end_to_end_simulation`, `plotting.plot_result` |

Defaults (`ProteusConfig`): θ = 10 m, n_p ≥ 3, ω = 15°, l = 20 m, intersection threshold
distance ⊔ = 15 m, agreement count c ≥ 2.

## Implementation notes

The paper's experiments ran in simulation, where every trace comes from a known map edge.
Real traces have no such link back to the map, so a working implementation has to settle
details that the condensed pseudocode leaves implicit. Where the paper gives two forms of a
step, both are available. Each choice below is a parameter, so the alternative can be
selected.

**Algorithm 1.** The pseudocode computes `G = Area(M)/(N_a·m)` and a cell size of `s = √m`,
then places agents at `M_0 + c + s/2` with a column counter that wraps. This is implemented
as a row-major grid of `s × s` cells, where agents are spread with a stride of `G` cells and
start at cell centres snapped to the nearest road node. Each agent's bounding box (Rule 2)
covers its share of the area. If `N_a·m > Area`, several agents share a cell. That sharing
is needed for verification, since a point needs at least 3 other volunteers; this is the
`redundancy` argument of `end_to_end_simulation`.

**Algorithm 2.**
* The sub-edge `E` grows over consecutive verified points, as described in Section 4.2.1.
  It is closed at the first unverified point, and also at the end of a trace, so an edge
  still open when the trace ends is kept. (In the condensed pseudocode, `E` is created
  inside the point loop at line 4.)
* The map-matching step from the text runs first: a point with no point of `M` within θ is
  discarded (`map_match=True`), and a discarded point also breaks the current edge.
  `snap_to_map` optionally moves kept points onto `M`.
* The threshold follows Algorithm 2, `n_p ≥ 3` (`min_support=3`). The text and Fig. 2
  phrase it as "more than three"; set `min_support=4` for that form.
* `n_p` is computed in one of three ways (`count_mode`):
  * `"points"`: `C_all − C_cur`, literally as in the pseudocode.
  * `"traces"`: distinct other traces.
  * `"volunteers"`: distinct other volunteers. This is the default for real data. It
    matches the requirement that an edge be collected by at least three different
    volunteers. Simulated traces have a fixed sampling rate, but real ones do not, and in
    `"points"` mode one densely sampled trace can verify a point by itself.
  * The name of any column, such as `collection`: distinct other values of that column.

**Combined map.** With real traces, the verified edges have to be merged into one graph
explicitly. They are inserted track by track: each track is smoothed lightly, and a point that lies within θ of
an existing edge running in a similar direction strengthens that edge instead of adding
geometry. Node positions converge to the mean of the matched points. Each road is stored
once, so the only short cycles left are the circuits volunteers walk around intersections
(Rule 4, Fig. 5). Crossing paths that do not connect, such as a foot-over-bridge (Fig. 4),
are not joined, because the direction test fails.

**Algorithms 3–4.** `A` is treated as a node of the combined map, and `con > 2` as its
degree. A cycle counts only if every node of the cycle lies within ⊔ of `A`. The handler
deletes the cycle, adds one node at the cycle's centroid, connects every outgoing edge
(`con`) to it, and repeats until nothing changes. That way a junction circled by many
volunteers collapses into one node. A collapsed cycle that ends up with degree 2 was a noise
loop on a single road: it is removed but not reported as an intersection.

**Algorithm 5.** Angles are undirected, because roads are walked both ways. "Minimal sized
points" means subdividing at θ/4 by default. A pair agrees when `|buffer| ≥ |P_a|/2`
(lines 13–16). `agreement_matrix` evaluates all pairs at once, measuring the
exact distance to segment `B` (the limit of a fine subdivision); a test checks it against
`agree`.

**Algorithm 6.**
* *Sub-edge length `l`.* The paper gives two forms: the smallest verified edge
  (Algorithm 6) and an upper bound of the largest one (Section 4.2.3). On real traces the
  smallest verified edge can be only a few metres long, so the pipeline default is a fixed
  l = 20 m. `subdivision_length` accepts `"smallest"`, `"largest"` or a number in metres,
  and `min_subdivision_length` sets a lower bound.
* *Argument order.* By default the verified edge plays `A`, as in line 6 (`agree(e', e)`).
  `agreement_reference="map"` makes the map sub-edge `A`, the ground-truth role in
  Algorithm 5's signature. That form is stricter at the ends of destroyed road sections.
* *Input `E`.* `E` is the bag of all verified edges, cut into pieces of length ≤ l. The
  rule "c ≥ 2" needs that redundancy, and the de-duplicated combined map would remove it.
  `segment_source="combined"` uses the edges of `M_C` instead.

**Section 4.3.** Road-heuristic validation against FHWA street-design rules remains future
work, as noted in Section 4.3.

**Simulation (Sec 5).**
* *The x-edge models.* In this simulator, the 2-, 3- and 5-edge models are *x independent
  volunteer collections* of every edge (cf. Fig. 7, "using 2 agents for verification"),
  verified with Algorithm 2 (n_p ≥ 3).
* *Counting support.* A volunteer's walk is one collection: adjacent edges from the same
  walk cannot verify each other. Every duplicated edge (Duplication Constant) is an extra
  re-collection.
* *Accuracy.* Accuracy is the share of ground-truth edges reproduced, meaning at least half
  of the edge lies within θ of verified points. Precision is reported too.
* *Deletion.* Deletion is applied once, to the ground truth: it is the disaster.
* *Similarity.* Exact graph edit distance is NP-hard (Section 5.5). The default
  similarity is `1 − GED/(|V₁|+|E₁|+|V₂|+|E₂|)`, with GED computed under a spatial
  one-to-one node matching. This is exact for that matching and scales to city-sized maps.
  `method="networkx"` runs NetworkX `optimize_graph_edit_distance` with a timeout instead.
* *Coverage time.* Map coverage time is the number of steps until every node has been
  visited by at least 3 different agents. Agents move one edge per step.

## Reproducing the paper

* **Table 3 from the published data.** `proteus anova data/edge_verification_results_raw_downtown.csv`
  reproduces every η² and partial η² in Table 3 to four decimals. The method is a type II
  ANOVA with every factor categorical and main effects only. `tests/test_system.py` checks
  this.
* **Everything else.** `python examples/reproduce_paper.py [--osm] [--full-grid] [--jobs N]`
  runs Tables 2, 4 and 5, a Table-3-style parameter study on freshly simulated data, and
  end-to-end runs in the style of Figs. 7–8. The results go to `results/`; see
  [`results/REPORT.md`](results/REPORT.md) for the run committed with this code.
  * Without `--osm`, the scripts use the generated Downtown and UH maps from ProteusData as
    stand-in road networks. These are fragmented Proteus *outputs*, not the OpenStreetMap
    ground truth used in the paper. With `--osm`, the original networks are downloaded with
    OSMnx.
  * The simulation reproduces the qualitative findings: accuracy rises steeply from the
    2-edge to the 3-edge model, the 3- and 5-edge models are close, and accuracy falls as
    jitter grows.
  * Absolute numbers and the ranking of the minor factors differ from the paper. The
    simulator here was rebuilt around the real-trace pipeline rather than carried over from
    the paper's experiments. Its noise model, verification threshold and accuracy measure
    are therefore not identical to the original setup. In particular, the Duplication
    Constant matters more here than in the paper, because duplicated edges are what can
    give the 2- and 3-edge models a third independent collection.

### Known limitations seen in the committed run

* **Deployment results on the stand-in maps.** The stand-in Downtown map breaks into
  hundreds of small pieces, so its largest connected component is tiny. Its Table-5 row in
  `results/REPORT.md` is therefore not meaningful. Use `--osm` or a synthetic grid for
  deployment studies.
* **Short destroyed sections can stay in the map.** A destroyed road section shorter than
  about 3θ (≈ 30 m at θ = 10 m) cannot be told apart from GPS noise around its two intact
  ends. Part of such a section often survives Algorithm 6; this is reported as
  `destroyed_length_kept_%`. Long destroyed sections are removed.
* **Low coverage means missing roads.** Roads that fewer than four distinct volunteers
  walked stay unverified (n_p ≥ 3 other volunteers). In the end-to-end runs, recall
  follows the 3-agent coverage closely.

## Tests

```bash
pytest -q
```

The tests cover each algorithm against small hand-built cases, check the vectorised
support counting against brute force, check `agreement_matrix` against `agree`, and run the
full pipeline, the experiments, the Table 3 reproduction and the CLI.

## License

MIT — see [LICENSE](LICENSE).
