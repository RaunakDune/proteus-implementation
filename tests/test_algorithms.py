import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from proteus import (LocalProjection, RoadMap, agree, agreement_matrix, combine_maps,
                     distribute_agents, handle_intersections, intersect, verify_traces)
from proteus.geo import (SegmentIndex, densify, point_segment_distance, segment_angle,
                         split_polyline_by_length)
from proteus.intersections import build_combined_map
from proteus.verification import VerifiedEdge, count_support

PROJ = LocalProjection(29.75, -95.36)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_projection_roundtrip():
    lon, lat = -95.3612, 29.7531
    x, y = PROJ.to_xy(lon, lat)
    lo, la = PROJ.to_lonlat(x, y)
    assert abs(lo - lon) < 1e-12 and abs(la - lat) < 1e-12


def test_point_segment_distance():
    d = point_segment_distance(np.array([[0, 5], [15, 0], [-3, 4]]),
                               np.array([0, 0]), np.array([10, 0]))
    assert np.allclose(d, [5, 5, 5])


def test_undirected_angle():
    a = np.array([[0, 0], [10, 0]])
    b = np.array([[10, 1], [0, 1]])       # same line, opposite direction
    assert segment_angle(a, b) == pytest.approx(0)
    c = np.array([[0, 0], [0, 10]])
    assert segment_angle(a, c) == pytest.approx(90)


def test_split_polyline_equal_pieces():
    pieces = split_polyline_by_length(np.array([[0, 0], [100, 0]]), 30)
    assert len(pieces) == 4
    assert all(abs(np.linalg.norm(p[-1] - p[0]) - 25) < 1e-9 for p in pieces)


def test_segment_index_exact():
    segs = np.array([[[0, 0], [100, 0]], [[0, 50], [100, 50]]])
    idx = SegmentIndex(segs, sample_spacing=20)
    seg, dist = idx.nearest(np.array([[50, 3], [50, 40], [50, 25.5]]), 10)
    assert seg.tolist() == [0, 1, -1]
    assert np.allclose(dist[:2], [3, 10])


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #
def test_algorithm1_one_agent_per_cell():
    rm = RoadMap.synthetic_grid(5, 5, 100)                  # 400 m x 400 m
    starts = distribute_agents(rm, 4, 200 * 200)            # four 200 m cells
    xy = np.array([s.xy for s in starts])
    x0, y0, _, _ = rm.bbox_xy()
    assert np.allclose(sorted(map(tuple, xy - [x0, y0])),
                       sorted([(100, 100), (300, 100), (100, 300), (300, 300)]))
    assert all(s.node in rm.graph for s in starts)


def test_algorithm1_redundancy_shares_cells():
    rm = RoadMap.synthetic_grid(5, 5, 100)
    area = rm.area_m2()
    starts = distribute_agents(rm, 9, 3 * area / 9)         # ~3 agents per cell
    from collections import Counter
    cells = Counter(tuple(np.round(s.xy, 3)) for s in starts)
    assert len(cells) == 4 and min(cells.values()) >= 2


# --------------------------------------------------------------------------- #
# Algorithm 2
# --------------------------------------------------------------------------- #
def _line_traces(n_volunteers, y_offsets=None, length=100, spacing=2.0):
    rows = []
    for v in range(n_volunteers):
        y = 0.0 if y_offsets is None else y_offsets[v]
        for k, x in enumerate(np.arange(0, length + 1e-9, spacing)):
            rows.append((v, 0, float(k), x, y))
    df = pd.DataFrame(rows, columns=["volunteer_id", "trace_id", "timestamp", "x", "y"])
    df["lon"], df["lat"] = PROJ.to_lonlat(df.x.values, df.y.values)
    return df


def test_count_support_modes_match_bruteforce():
    rng = np.random.default_rng(1)
    n = 600
    xy = rng.uniform(0, 80, (n, 2))
    tk = rng.integers(0, 40, n)
    vk = tk // 5
    el = rng.random(n) > 0.2
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)) <= 8
    d = d & el[None, :]
    pts = d.sum(1) - (d & (tk[None] == tk[:, None])).sum(1)
    trc = [len(set(tk[d[i]]) - {tk[i]}) for i in range(n)]
    vol = [len(set(vk[d[i]]) - {vk[i]}) for i in range(n)]
    for loop in (0, 10_000):
        assert (count_support(xy, tk, vk, 8, "points", el, max_label_loop=loop) == pts).all()
        assert (count_support(xy, tk, vk, 8, "traces", el, max_label_loop=loop) == trc).all()
        assert (count_support(xy, tk, vk, 8, "volunteers", el, max_label_loop=loop) == vol).all()


def test_verification_requires_three_other_traces():
    three = verify_traces(_line_traces(3), 5.0, count_mode="traces", projection=PROJ)
    assert three.edges == []                      # n_p = 2 < 3
    four = verify_traces(_line_traces(4), 5.0, count_mode="traces", projection=PROJ)
    assert len(four.edges) == 4 and all(len(e) == 51 for e in four.edges)


def test_verification_points_mode_is_literal_pseudocode():
    # with dense sampling a single other trace supplies >= 3 nearby points
    res = verify_traces(_line_traces(2), 5.0, count_mode="points", projection=PROJ)
    assert len(res.edges) == 2


def test_verification_splits_edges_at_unverified_points():
    df = _line_traces(5)
    # volunteer 0 strays 50 m off the road between x=40 and x=60
    m = (df.volunteer_id == 0) & df.x.between(40, 60)
    df.loc[m, "y"] = 50.0
    df["lon"], df["lat"] = PROJ.to_lonlat(df.x.values, df.y.values)
    res = verify_traces(df.drop(columns=["x", "y"]), 5.0, count_mode="volunteers", projection=PROJ)
    v0 = [e for e in res.edges if e.volunteer_id == 0]
    assert len(v0) == 2
    assert v0[0].points[:, 0].max() < 40 and v0[1].points[:, 0].min() > 60


def test_map_matching_discards_far_points():
    rm = RoadMap.from_segments(np.array([[[0, 0], [100, 0]]]), PROJ)
    df = _line_traces(5, y_offsets=[30, 30, 30, 30, 30])
    res = verify_traces(df, 10.0, road_map=rm, count_mode="volunteers")
    assert not res.points["map_ok"].any() and res.edges == []


# --------------------------------------------------------------------------- #
# Algorithms 3 + 4
# --------------------------------------------------------------------------- #
def _plus_with_circuit():
    """Four arms meeting at the origin plus a small square circuit around it."""
    g = nx.Graph()
    pos = {"c": (0, 0), "N": (0, 50), "S": (0, -50), "E": (50, 0), "W": (-50, 0),
           "n": (0, 5), "e": (5, 0), "s": (0, -5), "w": (-5, 0)}
    for k, p in pos.items():
        g.add_node(k, pos=np.array(p, dtype=float))
    g.add_edges_from([("N", "n"), ("S", "s"), ("E", "e"), ("W", "w"),
                      ("n", "e"), ("e", "s"), ("s", "w"), ("w", "n")])
    return g


def test_intersect_finds_circuit():
    g = _plus_with_circuit()
    cyc = intersect(g, "n", radius=10)
    assert cyc is not None and len(cyc) >= 3
    assert {x for e in cyc for x in e} <= {"n", "e", "s", "w"}
    assert intersect(g, "N", radius=10) is None          # degree 1


def test_intersect_ignores_large_cycles():
    g = _plus_with_circuit()
    assert intersect(g, "n", radius=3) is None           # cycle not within radius


def test_handle_intersections_collapses_to_junction():
    res = handle_intersections(_plus_with_circuit(), radius=10)
    g = res.graph
    assert len(res.intersections) == 1
    assert np.allclose(res.intersections[0], [0, 0], atol=1e-9)
    j = [n for n in g if g.nodes[n].get("intersection")][0]
    assert set(g.neighbors(j)) == {"N", "S", "E", "W"}
    assert nx.cycle_basis(g) == []


def test_crossing_without_circuit_is_not_an_intersection():
    # two perpendicular verified edges crossing (a foot-over-bridge, Fig. 4)
    a = VerifiedEdge(0, 0, np.column_stack([np.linspace(-50, 50, 41), np.zeros(41)]))
    b = VerifiedEdge(1, 0, np.column_stack([np.zeros(41), np.linspace(-50, 50, 41)]))
    g = build_combined_map([a, b], spacing=5, merge_radius=10)
    assert nx.number_connected_components(g) == 2
    assert handle_intersections(g, 15).intersections == []


def test_combined_map_stores_shared_road_once():
    rng = np.random.default_rng(0)
    edges = [VerifiedEdge(v, 0, np.column_stack([np.linspace(0, 200, 101), rng.normal(0, 1, 101)]))
             for v in range(5)]
    g = build_combined_map(edges, spacing=5, merge_radius=10)
    assert nx.number_connected_components(g) == 1
    assert nx.cycle_basis(g) == []
    total = sum(np.linalg.norm(g.nodes[u]["pos"] - g.nodes[v]["pos"]) for u, v in g.edges)
    assert total < 260


# --------------------------------------------------------------------------- #
# Algorithms 5 + 6
# --------------------------------------------------------------------------- #
def test_agree_angle_and_overlap():
    a = np.array([[0, 0], [20, 0]])
    assert agree(a, np.array([[0, 3], [20, 3]]), 5, 10)
    assert not agree(a, np.array([[0, 0], [20, 8]]), 5, 10)        # 21.8 deg
    assert agree(a, np.array([[9, 0], [40, 0]]), 5, 10)            # 70 % overlap
    assert not agree(a, np.array([[16, 0], [40, 0]]), 5, 10)       # 45 % overlap
    assert not agree(a, np.array([[0, 6], [20, 6]]), 5, 10)        # too far


def test_agreement_matrix_matches_agree():
    rng = np.random.default_rng(3)
    refs = rng.uniform(0, 60, (40, 2, 2))
    tests = rng.uniform(0, 60, (40, 2, 2))
    ri, sj = agreement_matrix(refs, tests, theta=8, omega=20, step=0.25)
    fast = set(zip(ri.tolist(), sj.tolist()))
    slow = {(i, j) for i in range(40) for j in range(40)
            if agree(refs[i], tests[j], 8, 20, step=0.25)}
    # the vectorised version measures to the exact segment, agree() to its
    # subdivision points: allow disagreement only on near-ties
    assert len(fast ^ slow) <= 2


def test_combine_maps_keeps_supported_and_deletes_unsupported():
    g = nx.Graph()
    pts = {0: (0, 0), 1: (100, 0), 2: (100, 100)}
    for k, (x, y) in pts.items():
        lon, lat = PROJ.to_lonlat(x, y)
        g.add_node(k, x=float(lon), y=float(lat))
    g.add_edges_from([(0, 1), (1, 2)])
    pre = RoadMap(g, PROJ)
    # two volunteers confirm the first road only
    verified = np.array([[[x, dy], [x + 20, dy]] for dy in (1.0, -1.5) for x in range(0, 100, 20)])
    res = combine_maps(pre, verified, theta=5, omega=10, subdivision_length=20)
    assert res.subdivided.graph.number_of_edges() == 10
    assert res.kept.sum() == 5
    segs = res.map.segments()
    assert np.allclose(segs[:, :, 1], 0, atol=1e-6)                 # only y = 0 road kept


def test_combine_maps_needs_two_agreeing_edges():
    g = nx.Graph()
    for k, (x, y) in {0: (0, 0), 1: (100, 0)}.items():
        lon, lat = PROJ.to_lonlat(x, y)
        g.add_node(k, x=float(lon), y=float(lat))
    g.add_edge(0, 1)
    pre = RoadMap(g, PROJ)
    one = np.array([[[x, 0], [x + 20, 0]] for x in range(0, 100, 20)])
    assert combine_maps(pre, one, 5, 10, subdivision_length=20).kept.sum() == 0
    assert combine_maps(pre, one, 5, 10, subdivision_length=20, min_agreement=1).kept.sum() == 5


def test_subdivision_length_strategies():
    pre = RoadMap.synthetic_grid(3, 3, 100)
    verified = np.array([[[0, 0], [12, 0]], [[0, 0], [40, 0]]])
    assert combine_maps(pre, verified, 5, 10, subdivision_length="smallest",
                        min_subdivision_length=1).subdivision_length == pytest.approx(12)
    assert combine_maps(pre, verified, 5, 10, subdivision_length="largest").subdivision_length == pytest.approx(40)
    assert combine_maps(pre, verified, 5, 10, subdivision_length="smallest",
                        min_subdivision_length=25).subdivision_length == pytest.approx(25)


def test_math_sanity():
    assert math.isclose(densify(np.array([[0, 0], [10, 0]]), 2.5).shape[0], 5)
