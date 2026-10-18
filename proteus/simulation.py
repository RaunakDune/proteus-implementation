"""Agent-based simulation of volunteers collecting GPS traces (Sec 5.6).

Volunteers move along the (post-disaster) road graph one edge per time step
and record noisy GPS fixes while collecting.  Two movement policies are
provided:

``"rules"``  the Proteus movement rules (Sec 4.1 and 5.6.1)
    1. minimum and maximum continuous data-collection period;
    2. remain within the assigned bounding box;
    3. move along the centre of the road (only GPS noise is added);
    4. at an intersection, walk a circuit around all its corners before
       choosing an exit, and visit every outgoing edge at least once;
    plus: prefer unexplored edges, then the least explored edge, and prefer
    continuing in the same direction over turning back.

``"random"`` an unweighted random walk (Sec 5.6.2).

The *map coverage time* (Sec 5.6.3) is the number of time steps until every
node of the graph has been visited by at least ``k`` (3) different agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assignment import AgentStart, BBox
from .geo import bearing_deg
from .roadmap import RoadMap


@dataclass
class GPSModel:
    """Simplified GPS error model (Sec 5.3): isotropic Gaussian noise."""

    sigma: float = 3.0               # std-dev of horizontal error per axis (m)
    sample_spacing: float = 5.0      # distance between consecutive fixes (m)
    speed: float = 1.4               # m/s, used for time-stamps
    outlier_prob: float = 0.0        # probability of a gross error
    outlier_sigma: float = 50.0      # std-dev of gross errors (m)
    dropout_prob: float = 0.0        # probability a fix is lost (urban canyon)

    def observe(self, xy: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Return noisy observations and a keep-mask for true positions ``xy``."""
        noisy = xy + rng.normal(0.0, self.sigma, size=xy.shape)
        if self.outlier_prob > 0:
            out = rng.random(len(xy)) < self.outlier_prob
            noisy[out] += rng.normal(0.0, self.outlier_sigma, size=(int(out.sum()), 2))
        keep = rng.random(len(xy)) >= self.dropout_prob
        return noisy, keep


@dataclass
class AgentRules:
    min_collection_steps: int = 30   # Rule 1
    max_collection_steps: int = 120  # Rule 1
    rest_steps: int = 0              # pause between collection periods
    intersection_circuit: bool = True  # Rule 4
    circuit_radius: float = 6.0      # metres from the junction centre
    shared_knowledge: bool = True    # agents see each other's explored edges
    bbox_margin: float = 0.0


@dataclass
class VolunteerAgent:
    agent_id: int
    node: object
    bbox: BBox
    prev: object = None
    collecting: bool = True
    period_left: int = 0
    rest_left: int = 0
    trace_id: int = 0
    clock: float = 0.0
    edge_counts: dict = field(default_factory=dict)
    taken_from: dict = field(default_factory=dict)


class Simulation:
    def __init__(self, road_map: RoadMap, starts: list[AgentStart], *, policy: str = "rules",
                 rules: AgentRules | None = None, gps: GPSModel | None = None, seed: int = 0):
        if policy not in ("rules", "random"):
            raise ValueError("policy must be 'rules' or 'random'")
        self.map = road_map
        self.g = road_map.graph
        self.pos = road_map.node_positions()
        self.policy = policy
        self.rules = rules or AgentRules()
        self.gps = gps or GPSModel()
        self.rng = np.random.default_rng(seed)
        self.edge_counts: dict = {}
        self.visitors: dict = {n: set() for n in self.g.nodes}
        self.t = 0
        self._rows: list[tuple] = []
        self.agents = []
        for s in starts:
            a = VolunteerAgent(s.agent_id, s.node, s.bbox)
            a.period_left = self._period()
            self.visitors[s.node].add(a.agent_id)
            self.agents.append(a)

    # ------------------------------------------------------------------ #
    def _period(self) -> int:
        r = self.rules
        return int(self.rng.integers(r.min_collection_steps, r.max_collection_steps + 1))

    def _in_box(self, node, box) -> bool:
        p = self.pos[node]
        m = self.rules.bbox_margin
        return box[0] - m <= p[0] <= box[2] + m and box[1] - m <= p[1] <= box[3] + m

    @staticmethod
    def _ekey(u, v):
        return (u, v) if str(u) <= str(v) else (v, u)

    def _choose(self, a: VolunteerAgent):
        nbrs = list(self.g.neighbors(a.node))
        if not nbrs:
            return None
        if self.policy == "random":
            return nbrs[int(self.rng.integers(len(nbrs)))]
        inside = [n for n in nbrs if self._in_box(n, a.bbox)]          # Rule 2
        cands = inside or nbrs
        counts = self.edge_counts if self.rules.shared_knowledge else a.edge_counts
        taken = a.taken_from.get(a.node, set())
        here = self.pos[a.node]
        in_heading = bearing_deg(self.pos[a.prev], here) if a.prev is not None else None
        scored = []
        for n in cands:
            not_yet = 0 if n not in taken else 1                         # Rule 4: every exit once
            explored = counts.get(self._ekey(a.node, n), 0)              # unexplored, then least explored
            if in_heading is None:
                turn = 0.0
            else:
                d = abs((bearing_deg(here, self.pos[n]) - in_heading + 180) % 360 - 180)
                turn = d                                                 # prefer same direction
            scored.append((not_yet if self.g.degree(a.node) > 2 else 0,
                           explored, turn, self.rng.random(), n))
        scored.sort(key=lambda s: s[:4])
        return scored[0][4]

    def _emit(self, a: VolunteerAgent, path: np.ndarray):
        """Record GPS fixes along a true path (polyline in metres)."""
        if len(path) < 2:
            return
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total <= 0:
            return
        n = max(1, int(np.ceil(total / self.gps.sample_spacing)))
        s = np.linspace(0.0, total, n + 1)[1:]
        true = np.stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])], 1)
        ts = a.clock + s / self.gps.speed
        a.clock += total / self.gps.speed
        if not a.collecting:
            return
        obs, keep = self.gps.observe(true, self.rng)
        for (x, y), t in zip(obs[keep], ts[keep]):
            self._rows.append((a.agent_id, a.trace_id, float(t), float(x), float(y)))

    def _circuit(self, node, incoming_from) -> np.ndarray:
        """Closed loop of radius r around a junction, starting on the incoming road."""
        c = self.pos[node]
        r = self.rules.circuit_radius
        start = np.deg2rad(bearing_deg(c, self.pos[incoming_from]))
        n = max(8, int(np.ceil(2 * np.pi * r / max(self.gps.sample_spacing, 0.5))))
        ang = start + np.linspace(0, 2 * np.pi, n + 1)
        ring = c + r * np.stack([np.cos(ang), np.sin(ang)], 1)
        return np.vstack([ring, c])

    # ------------------------------------------------------------------ #
    def step(self) -> None:
        self.t += 1
        for a in self.agents:
            # Rule 1: collection periods
            if a.collecting:
                a.period_left -= 1
                if a.period_left <= 0:
                    if self.rules.rest_steps > 0:
                        a.collecting = False
                        a.rest_left = self.rules.rest_steps
                    a.trace_id += 1
                    a.period_left = self._period()
            else:
                a.rest_left -= 1
                if a.rest_left <= 0:
                    a.collecting = True
            nxt = self._choose(a)
            if nxt is None:
                continue
            u = a.node
            path = np.vstack([self.pos[u], self.pos[nxt]])
            # Rule 4: circuit around the junction we are arriving at
            if (self.policy == "rules" and self.rules.intersection_circuit
                    and self.g.degree(nxt) > 2):
                circuit = self._circuit(nxt, u)
                d = self.pos[nxt] - self.pos[u]
                L = float(np.linalg.norm(d))
                r = self.rules.circuit_radius
                if L > r:
                    p_in = self.pos[nxt] - d / L * r
                    path = np.vstack([self.pos[u], p_in, circuit])
            self._emit(a, path)
            k = self._ekey(u, nxt)
            self.edge_counts[k] = self.edge_counts.get(k, 0) + 1
            a.edge_counts[k] = a.edge_counts.get(k, 0) + 1
            a.taken_from.setdefault(u, set()).add(nxt)
            a.prev, a.node = u, nxt
            if a.collecting:
                self.visitors[nxt].add(a.agent_id)

    def run(self, n_steps: int) -> "Simulation":
        for _ in range(n_steps):
            self.step()
        return self

    def coverage(self, k: int = 3, nodes=None) -> float:
        nodes = list(self.g.nodes) if nodes is None else list(nodes)
        if not nodes:
            return 1.0
        return float(np.mean([len(self.visitors[n]) >= k for n in nodes]))

    def run_until_coverage(self, k: int = 3, max_steps: int = 100_000, nodes=None,
                           check_every: int = 10) -> int | None:
        """Run until every node in ``nodes`` was visited by >= k distinct agents.

        Returns the map coverage time (steps), or ``None`` if ``max_steps`` was
        reached first.
        """
        nodes = list(self.g.nodes) if nodes is None else list(nodes)
        while self.t < max_steps:
            self.step()
            if self.t % check_every == 0 and self.coverage(k, nodes) >= 1.0:
                return self.t
        return None

    def traces(self) -> pd.DataFrame:
        """Collected traces as a DataFrame (volunteer_id, trace_id, timestamp, lat, lon, x, y)."""
        df = pd.DataFrame(self._rows, columns=["volunteer_id", "trace_id", "timestamp", "x", "y"])
        lon, lat = self.map.proj.to_lonlat(df["x"].to_numpy(), df["y"].to_numpy())
        df["lon"], df["lat"] = lon, lat
        return df[["volunteer_id", "trace_id", "timestamp", "lat", "lon", "x", "y"]]
