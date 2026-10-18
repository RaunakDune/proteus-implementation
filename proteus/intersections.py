"""Combined trace map, intersection detection and handling
(Sec 4.2.2, Algorithms 3 and 4).

Verified edges from Algorithm 2 are merged into one *combined map* ``M``
(a graph).  Volunteers are asked to walk a small circuit around every
intersection before choosing an exit (movement Rule 4, Fig. 5); in the combined
map that circuit appears as a short cycle.  Algorithm 3 finds such cycles,
Algorithm 4 collapses each into a single intersection node and connects every
outgoing edge to it.

Overlaps that are *not* intersections (Fig. 4 -- a foot-over-bridge crossing a
road, or two roads pushed together by noise) do not produce a small cycle, so
they are left untouched.  To make sure crossing traces are not fused into a
false junction while the combined map is built, a trace point only matches an
existing road when it is close *and* travels in a similar direction
(``heading_tolerance``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjection, local_headings, resample_uniform, split_polyline_by_length
from .roadmap import RoadMap
from .verification import VerifiedEdge


# --------------------------------------------------------------------------- #
# Combined map from verified edges
# --------------------------------------------------------------------------- #
def build_combined_map(edges: list[VerifiedEdge], *, spacing: float = 5.0,
                       merge_radius: float = 10.0, heading_tolerance: float = 45.0,
                       heading_baseline: float = 20.0, smooth_window: float = 10.0,
                       refine: bool = True) -> nx.Graph:
    """Combine all verified edges into one graph (the combined map ``M``).

    Verified edges are inserted one at a time (incremental track insertion):

    * each edge is smoothed with a moving average over ``smooth_window``
      metres and resampled every ``spacing`` metres;
    * a point *matches* the map when an existing graph edge lies within
      ``merge_radius`` and runs in a compatible direction (undirected angle
      <= ``heading_tolerance``, headings measured over ``heading_baseline``
      metres).  Matched stretches add no geometry -- they increase the
      traversal ``count`` of the matched edge;
    * unmatched stretches create new nodes, joined to each other and, where
      they branch off from or rejoin the map, to the nearest end node of the
      matched edge.

    * with ``refine`` each node is moved to the running mean of the points
      matched to it, so the geometry converges to the consensus centre line
      rather than staying on the first volunteer's track.

    Because a road walked by several volunteers is stored only once, the only
    short cycles left in ``M`` are those a volunteer actually walked: the
    circuits around intersections (Rule 4), which Algorithm 3 looks for.
    The direction test keeps crossing paths that do not connect (a
    foot-over-bridge over a road, Fig. 4) from being fused into a junction.

    Nodes carry ``pos`` (metres) and ``count``; edges carry ``count`` and
    ``volunteers``.
    """
    g = nx.Graph()
    cell = max(merge_radius, 1e-6)
    r = merge_radius
    grid: dict[tuple[int, int], list[int]] = {}
    eu: list[int] = []       # edge endpoints, in insertion order
    ev: list[int] = []
    eh: list[float] = []     # edge heading (deg)
    nx_: list[float] = []    # node coordinates
    ny_: list[float] = []

    def add_node(x, y):
        k = len(nx_)
        nx_.append(x)
        ny_.append(y)
        g.add_node(k, count=1)
        return k

    def add_edge(a, b, vol):
        if a == b:
            return
        if g.has_edge(a, b):
            g.edges[a, b]["count"] += 1
            g.edges[a, b]["volunteers"].add(vol)
            return
        k = len(eu)
        g.add_edge(a, b, count=1, volunteers={vol}, eid=k)
        eu.append(a)
        ev.append(b)
        eh.append(math.degrees(math.atan2(ny_[b] - ny_[a], nx_[b] - nx_[a])))
        x0, x1 = sorted((nx_[a], nx_[b]))
        y0, y1 = sorted((ny_[a], ny_[b]))
        for i in range(int(math.floor(x0 / cell)), int(math.floor(x1 / cell)) + 1):
            for j in range(int(math.floor(y0 / cell)), int(math.floor(y1 / cell)) + 1):
                grid.setdefault((i, j), []).append(k)

    def match(px, py, h, exclude_from):
        bx, by = int(math.floor(px / cell)), int(math.floor(py / cell))
        best, best_d = -1, math.inf
        seen = set()
        for i in (bx - 1, bx, bx + 1):
            for j in (by - 1, by, by + 1):
                for k in grid.get((i, j), ()):
                    if k in seen or k >= exclude_from:
                        continue
                    seen.add(k)
                    a = abs(eh[k] - h) % 180.0
                    if min(a, 180.0 - a) > heading_tolerance:
                        continue
                    ax, ay, bx_, by_ = nx_[eu[k]], ny_[eu[k]], nx_[ev[k]], ny_[ev[k]]
                    dx, dy = bx_ - ax, by_ - ay
                    L2 = dx * dx + dy * dy
                    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
                    qx, qy = ax + t * dx - px, ay + t * dy - py
                    d = qx * qx + qy * qy
                    if d <= r * r and d < best_d:
                        best, best_d = k, d
        return best

    window = max(1, int(round(heading_baseline / (2 * spacing))))
    recent = max(2, int(math.ceil(2 * r / spacing)) + 1)
    for e in edges:
        pts = resample_uniform(e.points, spacing) if e.length > spacing else e.points[[0, -1]]
        pts = smooth_polyline(pts, max(1, int(round(smooth_window / (2 * spacing)))))
        hs = local_headings(pts, window=window)
        prev = None                  # graph node standing for the previous point
        prev_matched = False
        own_edges: list[int] = []    # edges this track created (by insertion index)
        for (px, py), h in zip(pts.tolist(), hs.tolist()):
            # do not let the track match the edges it has just laid down
            excl = own_edges[-recent] if len(own_edges) >= recent else (own_edges[0] if own_edges else len(eu))
            k = match(px, py, h, exclude_from=excl if own_edges else len(eu))
            if k >= 0:
                a, b = eu[k], ev[k]
                da = (nx_[a] - px) ** 2 + (ny_[a] - py) ** 2
                db = (nx_[b] - px) ** 2 + (ny_[b] - py) ** 2
                anchor = a if da <= db else b
                g.edges[a, b]["count"] += 1
                g.edges[a, b]["volunteers"].add(e.volunteer_id)
                # refine: the anchor node moves to the mean of the points matched to it
                c = g.nodes[anchor]["count"] + 1
                g.nodes[anchor]["count"] = c
                if refine:
                    nx_[anchor] += (px - nx_[anchor]) / c
                    ny_[anchor] += (py - ny_[anchor]) / c
                if prev is not None and not prev_matched:
                    n0 = len(eu)
                    add_edge(prev, anchor, e.volunteer_id)          # rejoin the map
                    if len(eu) > n0:
                        own_edges.append(n0)
                prev, prev_matched = anchor, True
            else:
                n = add_node(px, py)
                if prev is not None:
                    n0 = len(eu)
                    add_edge(prev, n, e.volunteer_id)               # extend / branch off
                    if len(eu) > n0:
                        own_edges.append(n0)
                prev, prev_matched = n, False
    for k in g.nodes:
        g.nodes[k]["pos"] = np.array([nx_[k], ny_[k]])
    g.remove_nodes_from([k for k in list(g.nodes) if g.degree(k) == 0])
    return g


def smooth_polyline(points: np.ndarray, half_window: int) -> np.ndarray:
    """Moving-average smoothing of a polyline (end points kept)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 3 or half_window < 1:
        return points.copy()
    k = 2 * half_window + 1
    pad = np.vstack([np.repeat(points[:1], half_window, 0), points, np.repeat(points[-1:], half_window, 0)])
    c = np.cumsum(np.vstack([np.zeros((1, 2)), pad]), axis=0)
    out = (c[k:] - c[:-k]) / k
    out[0], out[-1] = points[0], points[-1]
    return out


# --------------------------------------------------------------------------- #
# Algorithm 3: intersection detection
# --------------------------------------------------------------------------- #
def intersect(graph: nx.Graph, node, radius: float, *, local_nodes=None,
              min_cycle_edges: int = 3) -> list[tuple] | None:
    """Algorithm 3 -- INTERSECT(A, E).

    ``con`` <- number of edges connected to ``A``; if ``con > 2``, look at each
    edge ``e`` connected to ``A``: if ``e`` closes a cycle with at least
    ``min_cycle_edges`` (3) edges, and every edge of that cycle lies within the
    intersection threshold distance ``radius`` (the paper's ``⊔``) of ``A``,
    return the list of cycle edges.  Otherwise return ``None``.
    """
    if node not in graph or graph.degree(node) <= 2:        # lines 2-3
        return None
    p0 = graph.nodes[node]["pos"]
    if local_nodes is None:
        local_nodes = [n for n, d in graph.nodes(data=True)
                       if np.hypot(*(d["pos"] - p0)) <= radius]
    local = set(n for n in local_nodes if n in graph)
    local.add(node)
    sub = nx.Graph(graph.subgraph(local))                   # every edge within radius
    for nbr in list(graph.neighbors(node)):                 # line 4
        if nbr not in local:
            continue
        sub.remove_edge(node, nbr)
        try:
            path = nx.shortest_path(sub, nbr, node)
        except nx.NetworkXNoPath:
            path = None
        sub.add_edge(node, nbr)
        if path is not None and len(path) >= min_cycle_edges:   # line 5
            cycle = [(node, nbr)] + list(zip(path[:-1], path[1:]))
            return cycle                                     # line 7
    return None                                              # line 12


# --------------------------------------------------------------------------- #
# Algorithm 4: intersection handler
# --------------------------------------------------------------------------- #
@dataclass
class IntersectionResult:
    graph: nx.Graph
    # positions (metres) of collapsed cycles that ended up as junctions
    # (degree > 2).  Cycles collapsing to a degree-2 node were noise loops on a
    # single road; they are removed but not reported as intersections.
    intersections: list[np.ndarray] = field(default_factory=list)
    cycles_collapsed: int = 0


def handle_intersections(graph: nx.Graph, radius: float, *, min_cycle_edges: int = 3,
                         max_passes: int = 20) -> IntersectionResult:
    """Algorithm 4 -- Intersection Handler.

    For every node of the combined map, call :func:`intersect`.  When a cycle
    ``E`` is found, collect ``con`` = all edges incident to cycle nodes that
    are not in ``E``, delete ``E`` from the map, and connect all edges in
    ``con`` to a single new intersection node placed at the centroid of the
    cycle.  Passes repeat until no more cycles are found, so a junction
    circled by several volunteers collapses into one node.
    """
    g = graph.copy()
    next_id = (max((n for n in g.nodes if isinstance(n, (int, np.integer))), default=-1) + 1)
    found: list[np.ndarray] = []
    for _ in range(max_passes):
        changed = False
        nodes = list(g.nodes)
        if not nodes:
            break
        pos = np.array([g.nodes[n]["pos"] for n in nodes])
        tree = cKDTree(pos)
        for i, n in enumerate(nodes):                         # line 1
            if n not in g or g.degree(n) <= 2:
                continue
            local = [nodes[j] for j in tree.query_ball_point(pos[i], radius)]
            cycle = intersect(g, n, radius, local_nodes=local,
                              min_cycle_edges=min_cycle_edges)   # line 2
            if cycle is None:                                 # line 4
                continue
            cyc_nodes = set(itertools.chain.from_iterable(cycle))
            con = []                                          # line 3
            for c in cyc_nodes:                               # lines 5-7
                for v in g.neighbors(c):
                    if v not in cyc_nodes:
                        con.append((c, v, dict(g.edges[c, v])))
            centre = np.mean([g.nodes[c]["pos"] for c in cyc_nodes], axis=0)
            g.remove_nodes_from(cyc_nodes)                    # line 9 (delete E)
            new = next_id
            next_id += 1
            g.add_node(new, pos=centre, count=1, heading=0.0, intersection=True)
            for _, v, data in con:                            # line 10
                if g.has_edge(new, v):
                    g.edges[new, v]["count"] = g.edges[new, v].get("count", 1) + data.get("count", 1)
                    g.edges[new, v].setdefault("volunteers", set()).update(data.get("volunteers", set()))
                else:
                    g.add_edge(new, v, **data)
            found.append(centre)
            changed = True
        if not changed:
            break
    g.remove_nodes_from([n for n in list(g.nodes) if g.degree(n) == 0])
    junctions = [g.nodes[n]["pos"] for n in g.nodes
                 if g.nodes[n].get("intersection") and g.degree(n) > 2]
    return IntersectionResult(graph=g, intersections=junctions,
                              cycles_collapsed=len(found))   # line 12


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #
def graph_chains(graph: nx.Graph) -> list[list]:
    """Decompose a graph into maximal chains of degree-2 nodes."""
    chains, seen = [], set()

    ends = [n for n in graph.nodes if graph.degree(n) != 2]
    for s in ends:
        for nbr in graph.neighbors(s):
            if (s, nbr) in seen or (nbr, s) in seen:
                continue
            chain = [s, nbr]
            seen.add((s, nbr))
            while graph.degree(chain[-1]) == 2:
                a, b = chain[-2], chain[-1]
                nxt = next(x for x in graph.neighbors(b) if x != a)
                if (b, nxt) in seen or (nxt, b) in seen:
                    break
                seen.add((b, nxt))
                chain.append(nxt)
            chains.append(chain)
    # pure cycles (all nodes degree 2)
    for u, v in graph.edges():
        if (u, v) in seen or (v, u) in seen:
            continue
        chain = [u, v]
        seen.add((u, v))
        while True:
            a, b = chain[-2], chain[-1]
            nxt = next((x for x in graph.neighbors(b) if x != a), None)
            if nxt is None or (b, nxt) in seen or (nxt, b) in seen:
                break
            seen.add((b, nxt))
            chain.append(nxt)
        chains.append(chain)
    return chains


def graph_to_segments(graph: nx.Graph, target_length: float) -> np.ndarray:
    """Turn a combined map into straight edges of length <= ``target_length``.

    Degree-2 chains are followed between junctions and cut into equal pieces;
    each piece becomes one straight edge (first point -> last point).  These
    are the "verified edges after intersection detection" fed to Algorithm 6.
    """
    segs = []
    for chain in graph_chains(graph):
        pts = np.array([graph.nodes[n]["pos"] for n in chain])
        for piece in split_polyline_by_length(pts, target_length):
            if np.hypot(*(piece[-1] - piece[0])) > 1e-6:
                segs.append([piece[0], piece[-1]])
    return np.array(segs).reshape(-1, 2, 2)


def verified_edges_to_segments(edges: list[VerifiedEdge], target_length: float, *,
                               spacing: float = 5.0, smooth_window: float = 10.0) -> np.ndarray:
    """The bag of verified edges as straight edges of length <= ``target_length``.

    Each verified edge is smoothed (as in :func:`build_combined_map`) and cut
    into equal pieces; each piece becomes one straight edge.  Unlike
    :func:`graph_to_segments` this keeps every volunteer's copy, which is the
    redundancy Algorithm 6 needs (a sub-edge survives only when >= 2 verified
    edges agree with it).
    """
    segs = []
    half = max(1, int(round(smooth_window / (2 * spacing))))
    for e in edges:
        pts = resample_uniform(e.points, spacing) if e.length > spacing else e.points[[0, -1]]
        pts = smooth_polyline(pts, half)
        for piece in split_polyline_by_length(pts, target_length):
            if np.hypot(*(piece[-1] - piece[0])) > 1e-6:
                segs.append([piece[0], piece[-1]])
    return np.array(segs).reshape(-1, 2, 2)


def graph_to_roadmap(graph: nx.Graph, projection: LocalProjection) -> RoadMap:
    g = nx.Graph()
    for n, d in graph.nodes(data=True):
        lon, lat = projection.to_lonlat(*d["pos"])
        g.add_node(n, x=float(lon), y=float(lat), **({"intersection": True} if d.get("intersection") else {}))
    for u, v, d in graph.edges(data=True):
        g.add_edge(u, v, count=d.get("count", 1))
    return RoadMap(g, projection)
