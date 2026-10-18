"""Volunteer data-collection model for the edge verification test (Sec 5.3).

Each simulated volunteer returns a noisy copy ``m_i`` of the ground-truth map
``M``, controlled by the five parameters of Table 1:

=====================  ==========================================  ========
Parameter              Effect                                      Default
=====================  ==========================================  ========
Jitter ``j``           displace 15% of edges (chosen at random) by   15 m
                       adding noise to their start and end points
Offset angle           maximum angular displacement: jittered and    10 deg
                       duplicated edges are rotated by up to this
                       angle; also the agreement tolerance
Duplication ``dc``     duplicate ``dc``% of edges at an angle <=     55 %
                       offset and a distance <= ``j`` from the
                       original
Split ``s``            split ``s``% of edges into sub-edges           35 %
Deletion ``d``         delete ``d``% of edges, creating gaps that     10 %
                       simulate obstructions
=====================  ==========================================  ========

Deletion is applied once to the ground truth (the disaster); the other
perturbations are drawn independently for every volunteer copy.  Every
collected point additionally receives small Gaussian GPS noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import densify, rotate_about
from .roadmap import RoadMap


@dataclass
class PerturbationParams:
    jitter: float = 15.0              # metres
    offset: float = 10.0              # degrees
    duplication: float = 55.0         # percent of edges
    split: float = 35.0               # percent of edges
    deletion: float = 10.0            # percent of edges
    jitter_fraction: float = 0.15     # fraction of edges that are jittered
    gps_sigma: float = 1.0            # per-point Gaussian noise (m)

    @classmethod
    def paper_defaults(cls) -> "PerturbationParams":
        return cls()


def apply_deletion(road_map: RoadMap, percent: float, rng: np.random.Generator) -> RoadMap:
    """Delete ``percent``% of edges at random (disaster obstructions)."""
    edges = road_map.edge_list()
    n = int(round(len(edges) * percent / 100.0))
    if n == 0:
        return road_map.copy()
    idx = rng.choice(len(edges), n, replace=False)
    return road_map.without_edges([edges[i] for i in idx])


def _random_unit(rng, n):
    a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([np.cos(a), np.sin(a)], 1)


def perturb_segments(segments: np.ndarray, p: PerturbationParams,
                     rng: np.random.Generator, return_info: bool = False):
    """One volunteer's noisy copy of ``segments`` (E, 2, 2 metres).

    Returns a list of polylines (each (n, 2)); split edges come back as
    several shorter polylines.  With ``return_info=True`` also returns, per
    polyline, the index of the source edge and whether it is a duplicate.
    """
    segs = np.asarray(segments, dtype=float).reshape(-1, 2, 2).copy()
    n = len(segs)
    if n == 0:
        return []
    # Jitter: displace a fraction of edges by ~j metres, rotated within offset
    k = int(round(n * p.jitter_fraction))
    jit = rng.choice(n, k, replace=False) if k else np.zeros(0, dtype=int)
    for i in jit:
        mag = np.abs(rng.normal(p.jitter, p.jitter / 4.0, size=2))
        segs[i, 0] += _random_unit(rng, 1)[0] * mag[0]
        segs[i, 1] += _random_unit(rng, 1)[0] * mag[1]
        if p.offset > 0:
            segs[i] = rotate_about(segs[i], segs[i].mean(0), rng.uniform(-p.offset, p.offset))
    out: list[np.ndarray] = []
    src: list[int] = []
    dup: list[bool] = []
    # Split
    ns = int(round(n * p.split / 100.0))
    split_set = set(rng.choice(n, ns, replace=False).tolist()) if ns else set()
    for i in range(n):
        if i in split_set:
            parts = int(rng.integers(2, 5))
            cuts = np.sort(rng.uniform(0.15, 0.85, parts - 1))
            t = np.concatenate([[0.0], cuts, [1.0]])
            for t0, t1 in zip(t[:-1], t[1:]):
                a = segs[i, 0] + t0 * (segs[i, 1] - segs[i, 0])
                b = segs[i, 0] + t1 * (segs[i, 1] - segs[i, 0])
                out.append(np.vstack([a, b]))
                src.append(i)
                dup.append(False)
        else:
            out.append(segs[i].copy())
            src.append(i)
            dup.append(False)
    # Duplication: copies at angle <= offset and distance <= j
    nd = int(round(n * p.duplication / 100.0))
    if nd:
        for i in rng.choice(n, nd, replace=False):
            s = segs[i].copy()
            s = rotate_about(s, s.mean(0), rng.uniform(-p.offset, p.offset))
            s = s + _random_unit(rng, 1)[0] * rng.uniform(0, p.jitter)
            out.append(s)
            src.append(int(i))
            dup.append(True)
    if return_info:
        return out, np.array(src, dtype=int), np.array(dup, dtype=bool)
    return out


def segments_to_traces(polylines: list[np.ndarray], volunteer_id, spacing: float,
                       gps_sigma: float, rng: np.random.Generator, start_trace: int = 0,
                       collections=None):
    """Sample each polyline every ``spacing`` metres (one trace per edge).

    ``collections`` optionally gives a collection label per polyline, written
    to a ``collection`` column (used to count independent collections).
    """
    rows = []
    t = 0.0
    for k, pl in enumerate(polylines):
        pts = densify(pl, spacing)
        pts = pts + rng.normal(0, gps_sigma, size=pts.shape) if gps_sigma > 0 else pts
        coll = collections[k] if collections is not None else f"{volunteer_id}"
        for x, y in pts:
            rows.append((volunteer_id, start_trace + k, t, float(x), float(y), coll))
            t += 1.0
    return rows
