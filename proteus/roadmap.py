"""Road-network container used as the pre-disaster map ``M`` and for outputs.

A :class:`RoadMap` wraps an undirected :class:`networkx.Graph` whose nodes
carry ``x`` (longitude) and ``y`` (latitude) attributes -- the same convention
OSMnx uses -- together with a :class:`~proteus.geo.LocalProjection` used to
convert to metres for all computations.

Edges are straight segments between their end nodes.  Curved roads are
represented as chains of short edges (load OSMnx graphs with
``simplify=False``, or call :meth:`RoadMap.subdivide`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import LocalProjection, split_polyline_by_length


class RoadMap:
    def __init__(self, graph: nx.Graph, projection: LocalProjection | None = None):
        if graph.is_directed():
            graph = nx.Graph(graph.to_undirected())
        if isinstance(graph, nx.MultiGraph):
            graph = nx.Graph(graph)
        self.graph: nx.Graph = graph
        if projection is None:
            if graph.number_of_nodes() == 0:
                projection = LocalProjection(0.0, 0.0)
            else:
                lons = [d["x"] for _, d in graph.nodes(data=True)]
                lats = [d["y"] for _, d in graph.nodes(data=True)]
                projection = LocalProjection.around(lons, lats)
        self.proj = projection

    # ------------------------------------------------------------------ #
    # Basic accessors
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (f"RoadMap(nodes={self.graph.number_of_nodes()}, "
                f"edges={self.graph.number_of_edges()}, "
                f"length={self.total_length():.0f} m)")

    def copy(self) -> "RoadMap":
        return RoadMap(self.graph.copy(), self.proj)

    @property
    def nodes(self) -> list:
        return list(self.graph.nodes)

    def node_xy(self, node) -> np.ndarray:
        d = self.graph.nodes[node]
        x, y = self.proj.to_xy(d["x"], d["y"])
        return np.array([float(x), float(y)])

    def node_positions(self) -> dict:
        """Map node -> (x, y) metres."""
        nodes = list(self.graph.nodes)
        if not nodes:
            return {}
        lon = np.array([self.graph.nodes[n]["x"] for n in nodes])
        lat = np.array([self.graph.nodes[n]["y"] for n in nodes])
        x, y = self.proj.to_xy(lon, lat)
        return {n: np.array([a, b]) for n, a, b in zip(nodes, x, y)}

    def edge_list(self) -> list[tuple]:
        return list(self.graph.edges())

    def segments(self) -> np.ndarray:
        """All edges as an array of shape (E, 2, 2) in metres."""
        pos = self.node_positions()
        edges = self.edge_list()
        if not edges:
            return np.zeros((0, 2, 2))
        return np.array([[pos[u], pos[v]] for u, v in edges])

    def edge_lengths(self) -> np.ndarray:
        s = self.segments()
        return np.linalg.norm(s[:, 1] - s[:, 0], axis=1) if len(s) else np.zeros(0)

    def total_length(self) -> float:
        return float(self.edge_lengths().sum())

    def bbox_xy(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in metres."""
        p = np.array(list(self.node_positions().values()))
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())

    def area_m2(self) -> float:
        x0, y0, x1, y1 = self.bbox_xy()
        return (x1 - x0) * (y1 - y0)

    def nearest_node(self, xy: np.ndarray, nodes: Iterable | None = None):
        pos = self.node_positions()
        keys = list(nodes) if nodes is not None else list(pos)
        arr = np.array([pos[k] for k in keys])
        _, i = cKDTree(arr).query(np.asarray(xy, dtype=float))
        if np.ndim(i) == 0:
            return keys[int(i)]
        return [keys[int(j)] for j in i]

    def largest_component(self) -> "RoadMap":
        if self.graph.number_of_nodes() == 0:
            return self.copy()
        cc = max(nx.connected_components(self.graph), key=len)
        return RoadMap(self.graph.subgraph(cc).copy(), self.proj)

    # ------------------------------------------------------------------ #
    # Transformations
    # ------------------------------------------------------------------ #
    def subdivide(self, max_length: float) -> "RoadMap":
        """Split every edge longer than ``max_length`` into equal sub-edges.

        This is line 2 of Algorithm 6 ("Sub-divide all edges in M to maximum
        length l").  Sub-edges stay connected through new intermediate nodes,
        whose ids are ``(u, v, k)`` tuples.  Each edge records its ``parent``
        edge in the original map.
        """
        g = nx.Graph()
        pos = self.node_positions()
        for n, d in self.graph.nodes(data=True):
            g.add_node(n, **d)
        for u, v in self.graph.edges():
            pieces = split_polyline_by_length(np.vstack([pos[u], pos[v]]), max_length)
            if len(pieces) == 1:
                g.add_edge(u, v, parent=(u, v))
                continue
            prev = u
            for k, piece in enumerate(pieces[:-1]):
                nid = ("sub", u, v, k)
                lon, lat = self.proj.to_lonlat(*piece[-1])
                g.add_node(nid, x=float(lon), y=float(lat))
                g.add_edge(prev, nid, parent=(u, v))
                prev = nid
            g.add_edge(prev, v, parent=(u, v))
        return RoadMap(g, self.proj)

    def with_edges(self, edges: Iterable[tuple]) -> "RoadMap":
        """New map containing only ``edges`` (and their end nodes)."""
        g = nx.Graph()
        for u, v in edges:
            g.add_node(u, **self.graph.nodes[u])
            g.add_node(v, **self.graph.nodes[v])
            g.add_edge(u, v, **self.graph.edges[u, v])
        return RoadMap(g, self.proj)

    def without_edges(self, edges: Iterable[tuple]) -> "RoadMap":
        g = self.graph.copy()
        g.remove_edges_from(list(edges))
        g.remove_nodes_from([n for n in list(g.nodes) if g.degree(n) == 0])
        return RoadMap(g, self.proj)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_segments(cls, segments_xy: np.ndarray, projection: LocalProjection,
                      snap_tolerance: float = 0.5) -> "RoadMap":
        """Build a graph from (E, 2, 2) metre segments, merging endpoints that
        are closer than ``snap_tolerance``."""
        segs = np.asarray(segments_xy, dtype=float).reshape(-1, 2, 2)
        g = nx.Graph()
        if len(segs) == 0:
            return cls(g, projection)
        ends = segs.reshape(-1, 2)
        tree = cKDTree(ends)
        parent = np.arange(len(ends))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in tree.query_pairs(snap_tolerance):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
        roots = np.array([find(i) for i in range(len(ends))])
        uniq, inv = np.unique(roots, return_inverse=True)
        centres = np.zeros((len(uniq), 2))
        np.add.at(centres, inv, ends)
        centres /= np.bincount(inv)[:, None]
        lon, lat = projection.to_lonlat(centres[:, 0], centres[:, 1])
        for k in range(len(uniq)):
            g.add_node(int(k), x=float(lon[k]), y=float(lat[k]))
        inv = inv.reshape(-1, 2)
        for a, b in inv:
            if a != b:
                g.add_edge(int(a), int(b))
        return cls(g, projection)

    @classmethod
    def from_edge_csv(cls, path: str | Path, snap_tolerance_m: float = 0.5) -> "RoadMap":
        """Load an edge list with columns ``x0, y0, x1, y1`` (lon/lat degrees).

        This is the format of ``houston_downtown_map_gen.csv`` and
        ``university_map_gen.csv`` in the ProteusData repository.
        """
        df = pd.read_csv(path)
        lon = np.concatenate([df.x0.values, df.x1.values])
        lat = np.concatenate([df.y0.values, df.y1.values])
        proj = LocalProjection.around(lon, lat)
        x0, y0 = proj.to_xy(df.x0.values, df.y0.values)
        x1, y1 = proj.to_xy(df.x1.values, df.y1.values)
        segs = np.stack([np.stack([x0, y0], 1), np.stack([x1, y1], 1)], 1)
        return cls.from_segments(segs, proj, snap_tolerance_m)

    def to_edge_csv(self, path: str | Path) -> None:
        rows = []
        for u, v in self.graph.edges():
            du, dv = self.graph.nodes[u], self.graph.nodes[v]
            rows.append((du["x"], du["y"], dv["x"], dv["y"]))
        pd.DataFrame(rows, columns=["x0", "y0", "x1", "y1"]).to_csv(path, index=False)

    @classmethod
    def from_osmnx(cls, center_lat: float, center_lon: float, dist_m: float = 1000,
                   network_type: str = "drive") -> "RoadMap":
        """Download a street network from OpenStreetMap with OSMnx (Sec 5.3).

        The paper used ``network_type='drive'`` for Downtown Houston
        (29.755733, -95.364688) and ``'walk'`` for the University of Houston
        campus (29.721489, -95.343374).  Requires ``pip install osmnx`` and
        network access to the Overpass API.
        """
        try:
            import osmnx as ox
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("RoadMap.from_osmnx needs the optional 'osmnx' package: "
                              "pip install osmnx") from e
        g = ox.graph_from_point((center_lat, center_lon), dist=dist_m,
                                network_type=network_type, simplify=False)
        g = nx.Graph(g.to_undirected())
        clean = nx.Graph()
        for n, d in g.nodes(data=True):
            clean.add_node(n, x=float(d["x"]), y=float(d["y"]))
        clean.add_edges_from((u, v) for u, v in g.edges() if u != v)
        return cls(clean)

    @classmethod
    def synthetic_grid(cls, rows: int = 8, cols: int = 8, spacing_m: float = 100.0,
                       center_lat: float = 29.755733, center_lon: float = -95.364688,
                       rotation_deg: float = 0.0, edge_subdivisions: int = 1) -> "RoadMap":
        """A regular street grid (like Downtown Houston) for tests and demos.

        ``edge_subdivisions`` > 1 splits each block into shorter edges, mimicking
        un-simplified OSM geometry.
        """
        proj = LocalProjection(center_lat, center_lon)
        t = np.deg2rad(rotation_deg)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        w, h = (cols - 1) * spacing_m, (rows - 1) * spacing_m
        g = nx.Graph()

        def add(nid, xy):
            xy = rot @ (np.asarray(xy) - [w / 2, h / 2])
            lon, lat = proj.to_lonlat(*xy)
            g.add_node(nid, x=float(lon), y=float(lat))

        for r in range(rows):
            for c in range(cols):
                add((r, c), (c * spacing_m, r * spacing_m))
        k = max(1, int(edge_subdivisions))
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((0, 1), (1, 0)):
                    r2, c2 = r + dr, c + dc
                    if r2 >= rows or c2 >= cols:
                        continue
                    prev = (r, c)
                    for i in range(1, k):
                        nid = (r, c, dr, dc, i)
                        f = i / k
                        add(nid, ((c + dc * f) * spacing_m, (r + dr * f) * spacing_m))
                        g.add_edge(prev, nid)
                        prev = nid
                    g.add_edge(prev, (r2, c2))
        return cls(g, proj)

    @classmethod
    def synthetic_campus(cls, size_m: float = 800.0, n_paths: int = 40, seed: int = 0,
                         center_lat: float = 29.721489, center_lon: float = -95.343374,
                         edge_length_m: float = 25.0) -> "RoadMap":
        """An irregular, curvy pedestrian network (like the UH campus).

        Built from a random geometric graph whose edges are bent into arcs and
        then chopped into ~``edge_length_m`` pieces.
        """
        rng = np.random.default_rng(seed)
        proj = LocalProjection(center_lat, center_lon)
        pts = rng.uniform(-size_m / 2, size_m / 2, size=(n_paths, 2))
        tri = _delaunay_edges(pts)
        # keep a sparse but connected subset: MST plus some extra edges
        full = nx.Graph()
        for i, j in tri:
            full.add_edge(i, j, weight=float(np.linalg.norm(pts[i] - pts[j])))
        keep = nx.minimum_spanning_tree(full)
        extra = [e for e in full.edges if not keep.has_edge(*e)]
        rng.shuffle(extra)
        for e in extra[: len(extra) // 2]:
            keep.add_edge(*e)
        segs = []
        for i, j in keep.edges():
            a, b = pts[i], pts[j]
            mid = (a + b) / 2
            normal = np.array([-(b - a)[1], (b - a)[0]])
            normal /= np.linalg.norm(normal) + 1e-12
            bend = rng.normal(0, 0.15) * np.linalg.norm(b - a)
            t = np.linspace(0, 1, 20)[:, None]
            ctrl = mid + bend * normal
            curve = (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b
            for piece in split_polyline_by_length(curve, edge_length_m):
                segs.append([piece[0], piece[-1]])
        return cls.from_segments(np.array(segs), proj, snap_tolerance=0.5)


def _delaunay_edges(points: np.ndarray) -> set[tuple[int, int]]:
    from scipy.spatial import Delaunay

    tri = Delaunay(points)
    edges = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = sorted((int(s[a]), int(s[b])))
            edges.add((i, j))
    return edges
