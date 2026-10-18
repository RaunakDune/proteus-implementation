"""Volunteer task assignment (Sec 4.1, Algorithm 1) and deployment strategies
used in the volunteer deployment experiment (Sec 5.6.3).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from .roadmap import RoadMap

BBox = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (metres)


@dataclass
class AgentStart:
    agent_id: int
    xy: np.ndarray          # starting location M_a (metres)
    node: object            # nearest road node to M_a
    bbox: BBox              # area-of-interest bounding box the agent must stay in (Rule 2)


# --------------------------------------------------------------------------- #
# Algorithm 1: Agent distribution in the AOI
# --------------------------------------------------------------------------- #
def distribute_agents(road_map: RoadMap, n_agents: int, max_area_per_agent: float,
                      bbox: BBox | None = None, first_id: int = 0) -> list[AgentStart]:
    """Algorithm 1 -- Agent Distribution in AOI.

    1. ``M`` <- bounding box of the pre-disaster map.
    2. The AOI is divided into regular square grid cells of side
       ``s = sqrt(m)``, where ``m`` is the maximum area allowed per agent.
    3. ``G = Area(M) / (N_a * m)`` is the number of cells available per agent;
       agents are laid out row-major over the grid with a stride of ``G`` cells
       using a cell counter ``c`` that wraps at the end of each row, and each
       agent starts at the centre of its cell (``M_0 + c + s/2``).

    The starting point is then snapped to the nearest road node so agents can
    begin collecting immediately.  Each agent's bounding box (Rule 2) is the
    square of area ``Area(M) / N_a`` (but at least its cell) centred on its
    start, clipped to ``M``.

    Parameters
    ----------
    road_map : pre-disaster map ``M``.
    n_agents : number of agents ``N_a``.
    max_area_per_agent : ``m`` in square metres.
    bbox : optionally restrict to a sub-box of the map (used for quadrants).
    """
    if n_agents <= 0:
        return []
    x0, y0, x1, y1 = bbox if bbox is not None else road_map.bbox_xy()
    width, height = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    area = width * height
    if n_agents * max_area_per_agent < area:
        warnings.warn(f"{n_agents} agents x {max_area_per_agent:.0f} m^2 cannot cover the "
                      f"{area:.0f} m^2 AOI; some cells will be unassigned.", stacklevel=2)
    s = math.sqrt(max_area_per_agent)                 # line 4
    cols = max(1, math.ceil(width / s))
    rows = max(1, math.ceil(height / s))
    n_cells = cols * rows
    cells_per_agent = max(1.0, area / (n_agents * max_area_per_agent))  # line 3 (G)
    cells_per_agent = min(cells_per_agent, n_cells / n_agents) if n_cells >= n_agents else 1.0

    # Nodes inside the box (fall back to all nodes)
    pos = road_map.node_positions()
    inside = [n for n, p in pos.items() if x0 <= p[0] <= x1 and y0 <= p[1] <= y1] or list(pos)

    side = max(s, math.sqrt(area / n_agents))
    starts: list[AgentStart] = []
    for i in range(n_agents):                          # line 6
        cell = int(math.floor(i * cells_per_agent)) % n_cells
        c, r = cell % cols, cell // cols               # column counter c wraps per row
        xy = np.array([x0 + c * s + s / 2, y0 + r * s + s / 2])   # line 7
        xy = np.minimum(xy, [x1, y1])
        node = road_map.nearest_node(xy, inside)
        box = (max(x0, xy[0] - side / 2), max(y0, xy[1] - side / 2),
               min(x1, xy[0] + side / 2), min(y1, xy[1] + side / 2))
        starts.append(AgentStart(first_id + i, xy, node, box))
    return starts


# --------------------------------------------------------------------------- #
# Deployment strategies from Sec 5.6.3
# --------------------------------------------------------------------------- #
def quadrant_bboxes(road_map: RoadMap, nx_: int = 2, ny_: int = 2, margin: float = 0.0) -> list[BBox]:
    x0, y0, x1, y1 = road_map.bbox_xy()
    xs = np.linspace(x0, x1, nx_ + 1)
    ys = np.linspace(y0, y1, ny_ + 1)
    out = []
    for j in range(ny_):
        for i in range(nx_):
            out.append((xs[i] - margin, ys[j] - margin, xs[i + 1] + margin, ys[j + 1] + margin))
    return out


def deploy_quadrants(road_map: RoadMap, agents_per_quadrant: int, n_split: int = 2,
                     margin: float = 75.0) -> list[AgentStart]:
    """Strategy 1: subdivide the AOI into quadrants, deploy a fixed number of
    agents into each (positions from Algorithm 1) and limit every agent to its
    quadrant's bounding box."""
    starts: list[AgentStart] = []
    for q in quadrant_bboxes(road_map, n_split, n_split, margin=0.0):
        area = (q[2] - q[0]) * (q[3] - q[1])
        m = area / agents_per_quadrant
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            part = distribute_agents(road_map, agents_per_quadrant, m, bbox=q, first_id=len(starts))
        qm = (q[0] - margin, q[1] - margin, q[2] + margin, q[3] + margin)
        for a in part:
            a.bbox = qm
        starts.extend(part)
    return starts


def poisson_disc_sample(bbox: BBox, radius: float, rng: np.random.Generator,
                        k: int = 30) -> np.ndarray:
    """Bridson's Poisson-disc sampling in a rectangle."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    cell = radius / math.sqrt(2)
    gw, gh = int(math.ceil(w / cell)) + 1, int(math.ceil(h / cell)) + 1
    grid = -np.ones((gw, gh), dtype=int)
    pts: list[np.ndarray] = []
    active: list[int] = []

    def fits(p):
        gx, gy = int(p[0] / cell), int(p[1] / cell)
        for i in range(max(0, gx - 2), min(gw, gx + 3)):
            for j in range(max(0, gy - 2), min(gh, gy + 3)):
                q = grid[i, j]
                if q >= 0 and np.sum((pts[q] - p) ** 2) < radius ** 2:
                    return False
        return True

    p0 = rng.uniform([0, 0], [w, h])
    pts.append(p0)
    grid[int(p0[0] / cell), int(p0[1] / cell)] = 0
    active.append(0)
    while active:
        idx = active[rng.integers(len(active))]
        found = False
        for _ in range(k):
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(radius, 2 * radius)
            p = pts[idx] + r * np.array([math.cos(ang), math.sin(ang)])
            if 0 <= p[0] < w and 0 <= p[1] < h and fits(p):
                pts.append(p)
                grid[int(p[0] / cell), int(p[1] / cell)] = len(pts) - 1
                active.append(len(pts) - 1)
                found = True
                break
        if not found:
            active.remove(idx)
    return np.array(pts) + [x0, y0]


def deploy_poisson(road_map: RoadMap, n_agents: int, seed: int = 0) -> list[AgentStart]:
    """Strategy 2: spread agents over the whole map with Poisson-disc sampling;
    agents are not limited to quadrants (bounding box = whole AOI)."""
    rng = np.random.default_rng(seed)
    bbox = road_map.bbox_xy()
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    radius = math.sqrt(area / (n_agents * 2.0))
    pts = poisson_disc_sample(bbox, radius, rng)
    while len(pts) < n_agents:
        radius *= 0.8
        pts = poisson_disc_sample(bbox, radius, rng)
    pts = pts[rng.choice(len(pts), n_agents, replace=False)]
    nodes = road_map.nearest_node(pts)
    return [AgentStart(i, p, n, bbox) for i, (p, n) in enumerate(zip(pts, nodes))]


def deploy_random(road_map: RoadMap, n_agents: int, seed: int = 0) -> list[AgentStart]:
    """Strategy 3: agents dropped on uniformly random road nodes."""
    rng = np.random.default_rng(seed)
    nodes = road_map.nodes
    bbox = road_map.bbox_xy()
    pos = road_map.node_positions()
    picks = rng.choice(len(nodes), n_agents, replace=True)
    return [AgentStart(i, pos[nodes[k]], nodes[k], bbox) for i, k in enumerate(picks)]
