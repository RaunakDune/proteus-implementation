"""Point-wise verification (Sec 4.2.1, Algorithm 2).

Input  : a bag of traces ``G`` (rows of ``<VolunteerID, lat, lon, timestamp>``).
Output : a bag of verified edges ``V``; each verified edge is a maximal run of
         consecutive points of one trace that were all verified.

Two steps (Sec 4.2.1):

1. *Map matching* -- every point is checked against the pre-disaster map
   ``M``; a point with no map location within the threshold distance ``theta``
   is discarded (treated as unverified, so it also breaks the current edge).
2. *Cross-trace verification* -- for every point ``p`` of trace ``g``::

       C_all = |points of G within theta of p|
       C_cur = |points of g within theta of p|
       n_p   = C_all - C_cur
       p is verified  iff  n_p >= k          (k = 3 in the paper)

   Consecutive verified points extend the current sub-edge ``E``; the first
   unverified point closes ``E`` and adds it to ``V``.

How ``n_p`` is counted is configurable through ``count_mode``:

``"points"``     literally ``C_all - C_cur`` (number of *points* from other
                 traces), exactly as written in the pseudocode.
``"traces"``     number of *distinct other traces* with a point within
                 ``theta`` (Sec 4.2.1: "how many points from different traces
                 are within the distance threshold").  Default.
``"volunteers"`` number of distinct *other volunteers* ("only edges that have
                 been collected by at least three different volunteers will be
                 considered verified").

any other string  the name of a column of the trace table; ``n_p`` is the
                 number of distinct *other* values of that column (e.g. a
                 "collection" id grouping several traces).

``"points"`` depends on the sampling rate (one dense trace can verify a point
by itself); ``"traces"`` and ``"volunteers"`` do not.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import SegmentIndex, polyline_length, project_onto_segment
from .roadmap import RoadMap
from .traces import global_trace_keys, order_traces, with_xy

COUNT_MODES = ("points", "traces", "volunteers")


@dataclass
class VerifiedEdge:
    volunteer_id: object
    trace_id: object
    points: np.ndarray                 # (n, 2) metres, in time order
    timestamps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    support: np.ndarray = field(default_factory=lambda: np.zeros(0))  # n_p per point

    def __len__(self) -> int:
        return len(self.points)

    @property
    def length(self) -> float:
        return polyline_length(self.points)


@dataclass
class VerificationResult:
    edges: list[VerifiedEdge]
    points: pd.DataFrame               # input points with x, y, n_p, map_ok, verified

    @property
    def verified_fraction(self) -> float:
        return float(self.points["verified"].mean()) if len(self.points) else 0.0


def count_support(xy: np.ndarray, trace_key: np.ndarray, volunteer_key: np.ndarray,
                  threshold: float, count_mode: str = "traces",
                  eligible: np.ndarray | None = None, chunk_size: int = 20_000,
                  max_label_loop: int = 64) -> np.ndarray:
    """Compute ``n_p`` for every point (lines 5-9 of Algorithm 2).

    ``eligible`` masks which points may act as *witnesses* (points discarded
    by map matching do not verify others).  With ``count_mode="labels"``,
    ``volunteer_key`` holds arbitrary integer labels whose distinct other
    values are counted.
    """
    if count_mode not in COUNT_MODES and count_mode != "labels":
        raise ValueError(f"count_mode must be one of {COUNT_MODES}")
    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=int)
    if eligible is None:
        eligible = np.ones(n, dtype=bool)
    wit_idx = np.flatnonzero(eligible)
    if len(wit_idx) == 0:
        return np.zeros(n, dtype=int)
    trace_key = np.asarray(trace_key, dtype=np.int64)
    volunteer_key = np.asarray(volunteer_key, dtype=np.int64)
    label = trace_key if count_mode in ("points", "traces") else volunteer_key
    n_labels = int(label.max()) + 1
    n_p = np.zeros(n, dtype=int)

    if count_mode == "points":
        # C_all: eligible points of all traces within theta (line 5)
        c_all = cKDTree(xy[wit_idx]).query_ball_point(xy, threshold, return_length=True)
        # C_cur: eligible points of the current trace within theta (line 7)
        c_cur = np.zeros(n, dtype=int)
        order = np.argsort(trace_key, kind="stable")
        bounds = np.flatnonzero(np.diff(trace_key[order])) + 1
        for grp in np.split(order, bounds):
            wit = grp[eligible[grp]]
            if len(wit):
                c_cur[grp] = cKDTree(xy[wit]).query_ball_point(xy[grp], threshold,
                                                               return_length=True)
        return np.asarray(c_all - c_cur, dtype=int)            # line 9

    if n_labels <= max_label_loop:
        # One KD-tree per trace/volunteer: n_p = number of *other* labels with
        # an eligible point within theta.  Cheap when labels are few.
        for lab in np.unique(label[wit_idx]):
            wit = wit_idx[label[wit_idx] == lab]
            d, _ = cKDTree(xy[wit]).query(xy, k=1, distance_upper_bound=threshold)
            n_p += (np.isfinite(d) & (label != lab)).astype(int)
        return n_p

    # Many labels: enumerate neighbour pairs in chunks.
    tree = cKDTree(xy[wit_idx])
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        hits = tree.query_ball_point(xy[start:stop], threshold, return_sorted=False)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        total = int(lens.sum())
        if total == 0:
            continue
        qi = np.repeat(np.arange(start, stop), lens)
        wj = wit_idx[np.fromiter(itertools.chain.from_iterable(hits), dtype=np.int64, count=total)]
        other = label[wj] != label[qi]
        qi, wj = qi[other], wj[other]
        key = np.unique(qi * n_labels + label[wj])
        n_p[start:stop] = np.bincount(key // n_labels - start, minlength=stop - start)
    return n_p


def verify_traces(traces: pd.DataFrame, threshold: float, *, road_map: RoadMap | None = None,
                  min_support: int = 3, count_mode: str = "traces", map_match: bool = True,
                  snap_to_map: bool = False, min_edge_points: int = 2,
                  projection=None) -> VerificationResult:
    """Algorithm 2 -- Trace Verification.

    Parameters
    ----------
    traces : DataFrame with ``volunteer_id, trace_id, timestamp, lat, lon``
        (``x``/``y`` metre columns are used directly if present).
    threshold : ``theta``, the maximum tolerated GPS noise in metres.
    road_map : the pre-disaster map ``M`` used for the map-matching step.
    min_support : a point is verified when ``n_p >= min_support`` (3 in the paper).
    count_mode : ``"points" | "traces" | "volunteers"`` or a column name
        (see module docstring).
    map_match : discard points with no point of ``M`` within ``theta``.
    snap_to_map : additionally replace each kept point by its closest point on
        ``M`` (off by default -- Proteus keeps the volunteer's location).
    min_edge_points : verified edges with fewer points are dropped (an edge
        needs at least two points).
    """
    if projection is None:
        projection = road_map.proj if road_map is not None else None
    df = traces
    if "x" not in df.columns or "y" not in df.columns:
        if projection is None:
            raise ValueError("need a projection or road_map to convert lat/lon to metres")
        df = with_xy(df, projection)
    df = order_traces(df)                                     # line 1
    xy = df[["x", "y"]].to_numpy(dtype=float)
    trace_key = global_trace_keys(df)
    if count_mode in COUNT_MODES:
        vol_key = pd.factorize(df["volunteer_id"])[0]
        mode = count_mode
    elif count_mode in df.columns:
        vol_key = pd.factorize(df[count_mode])[0]
        mode = "labels"
    else:
        raise ValueError(f"count_mode must be one of {COUNT_MODES} or a column name")

    # Step 1: map matching against the pre-disaster map M.
    map_ok = np.ones(len(df), dtype=bool)
    if map_match and road_map is not None and road_map.graph.number_of_edges():
        index = SegmentIndex(road_map.segments(), sample_spacing=max(threshold / 2, 1.0))
        seg, dist = index.nearest(xy, threshold)
        map_ok = seg >= 0
        if snap_to_map:
            ok = np.flatnonzero(map_ok)
            s = index.segments[seg[ok]]
            xy = xy.copy()
            xy[ok] = project_onto_segment(xy[ok], s[:, 0], s[:, 1])
            df["x"], df["y"] = xy[:, 0], xy[:, 1]

    # Step 2: cross-trace support.
    n_p = count_support(xy, trace_key, vol_key, threshold, mode, eligible=map_ok)
    verified = map_ok & (n_p >= min_support)                  # line 11
    df["n_p"] = n_p
    df["map_ok"] = map_ok
    df["verified"] = verified

    # Sequential sub-edge construction (lines 2-18).
    edges: list[VerifiedEdge] = []
    ts = df["timestamp"].to_numpy()
    vols = df["volunteer_id"].to_numpy()
    tids = df["trace_id"].to_numpy()
    bounds = np.flatnonzero(np.diff(trace_key) != 0) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(df)]])
    for s, e in zip(starts, ends):                            # for all traces g
        current: list[int] = []                               # empty sub-edge E
        for i in range(s, e):                                 # for all points p
            if verified[i]:
                current.append(i)                             # extend E with p
            else:
                if len(current) >= min_edge_points:
                    edges.append(_make_edge(current, xy, ts, vols, tids, n_p))  # add E to V
                current = []                                  # new empty E
        if len(current) >= min_edge_points:                   # close E at trace end
            edges.append(_make_edge(current, xy, ts, vols, tids, n_p))
    return VerificationResult(edges=edges, points=df)


def _make_edge(idx, xy, ts, vols, tids, n_p) -> VerifiedEdge:
    idx = np.asarray(idx)
    return VerifiedEdge(volunteer_id=vols[idx[0]], trace_id=tids[idx[0]],
                        points=xy[idx].copy(), timestamps=ts[idx].copy(),
                        support=n_p[idx].copy())


def verified_edges_to_frame(edges: list[VerifiedEdge], projection) -> pd.DataFrame:
    """Flatten verified edges into a long table (edge_id, seq, lon, lat, ...)."""
    rows = []
    for k, e in enumerate(edges):
        lon, lat = projection.to_lonlat(e.points[:, 0], e.points[:, 1])
        for j in range(len(e)):
            rows.append((k, j, e.volunteer_id, e.trace_id,
                         e.timestamps[j] if len(e.timestamps) else np.nan, lon[j], lat[j]))
    return pd.DataFrame(rows, columns=["edge_id", "seq", "volunteer_id", "trace_id",
                                       "timestamp", "lon", "lat"])
