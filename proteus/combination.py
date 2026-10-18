"""Path combination (Sec 4.2.3, Algorithms 5 and 6).

Algorithm 5 (``AGREE``) decides whether two edges agree: they must be aligned
(angle within the tolerance ``omega``) and overlap -- at least half of the
points obtained by subdividing ``A`` lie within ``theta`` of ``B``.

Algorithm 6 combines the verified edges with the pre-disaster map ``M``:
``M`` is subdivided into sub-edges no longer than ``l``; a sub-edge survives
only when at least two verified edges agree with it.  The surviving,
connected sub-edges form the combined post-disaster map.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .geo import SegmentIndex, densify, segment_angle
from .roadmap import RoadMap


# --------------------------------------------------------------------------- #
# Algorithm 5: agreement function
# --------------------------------------------------------------------------- #
def _as_segment(e) -> np.ndarray:
    e = np.asarray(e, dtype=float)
    return np.vstack([e[0], e[-1]])


def agree(a, b, theta: float, omega: float, step: float | None = None) -> bool:
    """Algorithm 5 -- AGREE(A, B, theta, omega).

    ``a`` is the reference edge, ``b`` the edge being tested; each is a
    polyline of shape (n, 2) in metres (a straight edge is just its two end
    points).  The edge direction is taken from first to last point.

    1. if angle(A, B) > omega: return False
    2. P_a, P_b <- A and B subdivided into minimal sized points (``step``,
       default ``theta / 4``)
    3. buffer <- points of P_a whose minimum distance to P_b is < theta
    4. return |buffer| >= |P_a| / 2
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if segment_angle(_as_segment(a), _as_segment(b)) > omega:   # lines 2-4
        return False
    step = theta / 4.0 if step is None else step
    pa = densify(a, step)                                          # line 5
    pb = densify(b, step)                                          # line 6
    dmin, _ = cKDTree(pb).query(pa)                                # line 9
    buffer = int(np.count_nonzero(dmin < theta))                   # lines 7-12
    return not buffer < len(pa) / 2.0                              # lines 13-16


def agreement_matrix(refs: np.ndarray, tests: np.ndarray, theta: float, omega: float,
                     step: float | None = None):
    """Vectorised AGREE for many pairs.

    Evaluates ``agree(refs[i], tests[j])`` for every pair that could possibly
    agree and returns the agreeing index pairs ``(i, j)``.  ``refs`` and
    ``tests`` are straight edges shaped (n, 2, 2).  The distance from a point
    of ``P_a`` to the subdivided ``P_b`` is computed exactly as the distance
    to segment ``B`` (the limit of a fine subdivision).
    """
    refs = np.asarray(refs, dtype=float).reshape(-1, 2, 2)
    tests = np.asarray(tests, dtype=float).reshape(-1, 2, 2)
    if len(refs) == 0 or len(tests) == 0:
        e = np.zeros(0, dtype=int)
        return e, e
    step = theta / 4.0 if step is None else step
    pts, owner = [], []
    for i, s in enumerate(refs):
        d = densify(s, step)
        pts.append(d)
        owner.append(np.full(len(d), i))
    pts = np.vstack(pts)
    owner = np.concatenate(owner)
    n_pa = np.bincount(owner, minlength=len(refs))
    index = SegmentIndex(tests, sample_spacing=max(step, 1.0))
    pi, sj, d = index.query_pairs(pts, theta)
    keep = d < theta
    pi, sj = pi[keep], sj[keep]
    if len(pi) == 0:
        e = np.zeros(0, dtype=int)
        return e, e
    ri = owner[pi]
    key = ri.astype(np.int64) * len(tests) + sj
    uk, cnt = np.unique(key, return_counts=True)
    ri, sj = uk // len(tests), uk % len(tests)
    ok = cnt >= n_pa[ri] / 2.0
    ri, sj = ri[ok], sj[ok]
    ang_ok = segment_angle(refs[ri], tests[sj]) <= omega
    return ri[ang_ok], sj[ang_ok]


# --------------------------------------------------------------------------- #
# Algorithm 6: map combination
# --------------------------------------------------------------------------- #
@dataclass
class CombinationResult:
    map: RoadMap                  # combined map M
    subdivided: RoadMap           # M' before deletion
    support: np.ndarray           # agreeing verified edges per sub-edge of M'
    kept: np.ndarray              # boolean mask over M' edges
    subdivision_length: float


def resolve_subdivision_length(verified: np.ndarray, subdivision_length, minimum: float) -> float:
    lengths = np.linalg.norm(verified[:, 1] - verified[:, 0], axis=1) if len(verified) else np.zeros(1)
    if subdivision_length == "smallest":       # Algorithm 6, line 1
        l = float(lengths.min())
    elif subdivision_length == "largest":      # Sec 4.2.3 text
        l = float(lengths.max())
    else:
        l = float(subdivision_length)
    return max(l, minimum)


def combine_maps(pre_map: RoadMap, verified: np.ndarray, theta: float, omega: float, *,
                 subdivision_length="smallest", min_subdivision_length: float = 10.0,
                 min_agreement: int = 2, min_component_edges: int = 1,
                 reference: str = "verified", step: float | None = None) -> CombinationResult:
    """Algorithm 6 -- Map Combination.

    Parameters
    ----------
    pre_map : pre-disaster map ``M``.
    verified : verified edges after intersection detection ``E`` as an
        (n, 2, 2) array of straight edges in metres.
    theta, omega : threshold distance and tolerance angle for :func:`agree`.
    subdivision_length : ``"smallest"`` (pseudocode line 1: length of the
        smallest edge in ``E``), ``"largest"`` (Sec 4.2.3 text: sub-edges at
        most the size of the largest verified edge) or a number of metres.
    min_subdivision_length : floor on ``l`` so a tiny verified edge cannot
        explode the subdivision.
    min_agreement : a sub-edge is deleted when fewer than this many verified
        edges agree with it (2 in the paper).
    min_component_edges : connected components of the result with fewer edges
        are dropped (1 keeps everything that is connected to anything).
    reference : which edge plays ``A`` (the edge that is subdivided and must
        lie mostly within ``theta`` of the other).  ``"verified"`` follows
        Algorithm 6 line 6, ``agree(e', e)``; ``"map"`` follows the signature of
        Algorithm 5, where ``A`` is the ground-truth (map) edge -- a sub-edge then
        survives only if most of *its own* length is covered by verified edges,
        which is stricter at the ends of destroyed road sections.
    """
    verified = np.asarray(verified, dtype=float).reshape(-1, 2, 2)
    l = resolve_subdivision_length(verified, subdivision_length, min_subdivision_length)
    sub = pre_map.subdivide(l)                                     # line 2
    sub_edges = sub.edge_list()
    sub_segs = sub.segments()
    # lines 3-9: c(e) = #{e' in E : agree(e', e, theta, omega)}
    if reference == "verified":
        _, sj = agreement_matrix(verified, sub_segs, theta, omega, step=step)
    elif reference == "map":
        sj, _ = agreement_matrix(sub_segs, verified, theta, omega, step=step)
    else:
        raise ValueError("reference must be 'verified' or 'map'")
    support = np.bincount(sj, minlength=len(sub_edges))
    kept = support >= min_agreement                                # lines 10-12
    g = nx.Graph()
    for (u, v), k in zip(sub_edges, kept):
        if k:
            g.add_node(u, **sub.graph.nodes[u])
            g.add_node(v, **sub.graph.nodes[v])
            g.add_edge(u, v, **sub.graph.edges[u, v])
    if min_component_edges > 1:                                    # line 14
        for cc in list(nx.connected_components(g)):
            if g.subgraph(cc).number_of_edges() < min_component_edges:
                g.remove_nodes_from(cc)
    return CombinationResult(map=RoadMap(g, pre_map.proj), subdivided=sub,
                             support=support, kept=kept, subdivision_length=l)
