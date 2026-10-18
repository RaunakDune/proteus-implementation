"""Minimal end-to-end example.

1. build (or load) a pre-disaster road map,
2. destroy 10 % of the roads,
3. let 24 simulated volunteers collect GPS traces on what is left,
4. run Proteus on the traces and score the result.

    python examples/quickstart.py
"""

import numpy as np

from proteus import (GPSModel, Proteus, ProteusConfig, RoadMap, Simulation, apply_deletion,
                     deploy_quadrants, map_metrics)

# 1. Pre-disaster map M.  For a real area use e.g.
#    RoadMap.from_osmnx(29.755733, -95.364688, dist_m=800, network_type="drive")
pre_map = RoadMap.synthetic_grid(rows=6, cols=6, spacing_m=100, edge_subdivisions=2)

# 2. The disaster: 10 % of road segments become impassable.
post_map = apply_deletion(pre_map, 10, np.random.default_rng(0))

# 3. Volunteers (Sec 4.1 rules) collect noisy GPS traces on the post-disaster network.
starts = deploy_quadrants(post_map, agents_per_quadrant=6)
sim = Simulation(post_map, starts, gps=GPSModel(sigma=3.0), seed=0).run(800)
traces = sim.traces()          # volunteer_id, trace_id, timestamp, lat, lon (+ x, y)
print(f"{len(traces)} GPS fixes from {traces.volunteer_id.nunique()} volunteers")

# 4. Proteus: verification (Alg 2) -> intersections (Alg 3/4) -> combination (Alg 5/6)
result = Proteus(pre_map, ProteusConfig(threshold=10.0)).run(traces)
print(result.summary())
print(map_metrics(post_map, result.final_map, theta=10.0))

result.final_map.to_edge_csv("post_disaster_map.csv")
