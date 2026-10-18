"""Geometry helpers.

All Proteus algorithms work in a *local metric* frame (metres east / north of a
reference point).  City-scale areas of interest (a few km across) are small
enough that an equirectangular projection around the area's centroid is
accurate to well under a metre, which is far below GPS noise.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_008.8


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LocalProjection:
    """Equirectangular projection centred on ``(lat0, lon0)``.

    ``to_xy`` maps (lon, lat) in degrees to (x, y) metres; ``to_lonlat`` is the
    inverse.  Both accept scalars or numpy arrays.
    """

    lat0: float
    lon0: float

    @property
    def _kx(self) -> float:
        return np.deg2rad(1.0) * EARTH_RADIUS_M * np.cos(np.deg2rad(self.lat0))

    @property
    def _ky(self) -> float:
        return np.deg2rad(1.0) * EARTH_RADIUS_M

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return (lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky

    def to_lonlat(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return x / self._kx + self.lon0, y / self._ky + self.lat0

    def points_to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=float)
        x, y = self.to_xy(lonlat[..., 0], lonlat[..., 1])
        return np.stack([x, y], axis=-1)

    def points_to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        lon, lat = self.to_lonlat(xy[..., 0], xy[..., 1])
        return np.stack([lon, lat], axis=-1)

    @classmethod
    def around(cls, lon, lat) -> "LocalProjection":
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return cls(lat0=float(np.mean(lat)), lon0=float(np.mean(lon)))


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres."""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# Angles
# --------------------------------------------------------------------------- #
def bearing_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Heading of the vector a->b in degrees, measured CCW from +x (east)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    return np.degrees(np.arctan2(d[..., 1], d[..., 0]))


def undirected_angle_diff(h1, h2) -> np.ndarray:
    """Smallest angle between two *undirected* lines with headings h1, h2.

    Roads are traversed in both directions, so a trace heading north agrees
    with a road segment digitised south-bound.  Result is in [0, 90].
    """
    d = np.abs(np.asarray(h1, dtype=float) - np.asarray(h2, dtype=float)) % 180.0
    return np.minimum(d, 180.0 - d)


def segment_angle(seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Undirected angle (deg) between segments shaped (..., 2, 2)."""
    seg_a = np.asarray(seg_a, dtype=float)
    seg_b = np.asarray(seg_b, dtype=float)
    return undirected_angle_diff(bearing_deg(seg_a[..., 0, :], seg_a[..., 1, :]),
                                 bearing_deg(seg_b[..., 0, :], seg_b[..., 1, :]))


def rotate_about(points: np.ndarray, centre: np.ndarray, angle_deg: float) -> np.ndarray:
    t = np.deg2rad(angle_deg)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (np.asarray(points) - centre) @ r.T + centre


# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #
def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise distance from points ``p`` to segments ``a``-``b``.

    All inputs have shape (..., 2) and broadcast together.
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = np.einsum("...i,...i->...", ab, ab)
    t = np.where(denom > 0,
                 np.einsum("...i,...i->...", p - a, ab) / np.where(denom > 0, denom, 1.0),
                 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def project_onto_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closest point on segment a-b to p (element-wise)."""
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    ab = b - a
    denom = np.einsum("...i,...i->...", ab, ab)
    t = np.where(denom > 0,
                 np.einsum("...i,...i->...", p - a, ab) / np.where(denom > 0, denom, 1.0),
                 0.0)
    t = np.clip(t, 0.0, 1.0)
    return a + t[..., None] * ab


# --------------------------------------------------------------------------- #
# Polylines
# --------------------------------------------------------------------------- #
def polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def densify(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline so consecutive vertices are at most ``spacing`` apart.

    Original vertices are kept; intermediate points are inserted uniformly on
    each segment.  This is the "subdivide into minimal sized points" step used
    by the agreement function (Algorithm 5).
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2 or spacing <= 0:
        return points.copy()
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        seg_len = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(seg_len / spacing)))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


def resample_uniform(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at (approximately) uniform arc-length ``spacing``.

    Unlike :func:`densify` this does *not* keep the original vertices, so the
    output has evenly spaced points; the first and last points are kept.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points.copy()
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return points[:1].copy()
    n = max(1, int(round(total / spacing)))
    s = np.linspace(0.0, total, n + 1)
    x = np.interp(s, cum, points[:, 0])
    y = np.interp(s, cum, points[:, 1])
    return np.stack([x, y], axis=1)


def split_polyline_by_length(points: np.ndarray, max_len: float) -> list[np.ndarray]:
    """Cut a polyline into consecutive pieces of length <= ``max_len``.

    Pieces are of equal length (total / ceil(total / max_len)) so a polyline is
    never left with a tiny remainder piece.
    """
    points = np.asarray(points, dtype=float)
    total = polyline_length(points)
    if total <= max_len * (1 + 1e-9) or total == 0:
        return [points.copy()]
    n = int(np.ceil(total / max_len - 1e-9))
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    cuts = np.linspace(0.0, total, n + 1)
    pieces = []
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        inner = (cum > s0) & (cum < s1)
        start = np.array([np.interp(s0, cum, points[:, 0]), np.interp(s0, cum, points[:, 1])])
        end = np.array([np.interp(s1, cum, points[:, 0]), np.interp(s1, cum, points[:, 1])])
        pieces.append(np.vstack([start, points[inner], end]))
    return pieces


def local_headings(points: np.ndarray, window: int = 1) -> np.ndarray:
    """Per-vertex heading (deg) of a polyline using central differences over
    ``window`` vertices on each side (a wider window suppresses GPS noise)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        return np.zeros(n)
    w = max(1, int(window))
    prev_idx = np.clip(np.arange(n) - w, 0, n - 1)
    next_idx = np.clip(np.arange(n) + w, 0, n - 1)
    return bearing_deg(points[prev_idx], points[next_idx])


# --------------------------------------------------------------------------- #
# Spatial index over line segments
# --------------------------------------------------------------------------- #
class SegmentIndex:
    """Radius queries from points to a set of line segments.

    Segments are densified into sample points held in a KD-tree; candidate
    segments are found via the tree and the *exact* point-to-segment distance
    is then computed for each candidate, so results are exact.
    """

    def __init__(self, segments: np.ndarray, sample_spacing: float = 5.0):
        segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
        self.segments = segments
        self.sample_spacing = float(sample_spacing)
        pts, ids = [], []
        for i, (a, b) in enumerate(segments):
            d = densify(np.vstack([a, b]), sample_spacing)
            pts.append(d)
            ids.append(np.full(len(d), i))
        if pts:
            self._pts = np.vstack(pts)
            self._ids = np.concatenate(ids)
        else:
            self._pts = np.zeros((0, 2))
            self._ids = np.zeros(0, dtype=int)
        self._tree = cKDTree(self._pts) if len(self._pts) else None

    def __len__(self) -> int:
        return len(self.segments)

    def query_pairs(self, points: np.ndarray, radius: float):
        """Return ``(point_idx, seg_idx, distance)`` for every pair within ``radius``."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if self._tree is None or len(points) == 0:
            e = np.zeros(0, dtype=int)
            return e, e, np.zeros(0)
        # Any point within `radius` of a segment is within radius + spacing/2
        # of one of that segment's sample points.
        r = radius + self.sample_spacing / 2.0 + 1e-9
        hits = self._tree.query_ball_point(points, r, return_sorted=False)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        total = int(lens.sum())
        if total == 0:
            e = np.zeros(0, dtype=int)
            return e, e, np.zeros(0)
        pi = np.repeat(np.arange(len(points), dtype=np.int64), lens)
        si = self._ids[np.fromiter(itertools.chain.from_iterable(hits), dtype=np.int64, count=total)]
        n_seg = max(len(self.segments), 1)
        key = np.unique(pi * n_seg + si)
        pi, si = key // n_seg, key % n_seg
        d = point_segment_distance(points[pi], self.segments[si, 0], self.segments[si, 1])
        keep = d <= radius
        return pi[keep], si[keep], d[keep]

    def nearest(self, points: np.ndarray, max_radius: float):
        """Nearest segment within ``max_radius`` for each point.

        Returns ``(seg_idx, distance)``; seg_idx is -1 where none is in range.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        seg = np.full(len(points), -1, dtype=int)
        dist = np.full(len(points), np.inf)
        pi, si, d = self.query_pairs(points, max_radius)
        if len(pi):
            order = np.lexsort((d, pi))
            pi, si, d = pi[order], si[order], d[order]
            first = np.concatenate([[True], pi[1:] != pi[:-1]])
            seg[pi[first]] = si[first]
            dist[pi[first]] = d[first]
        return seg, dist
