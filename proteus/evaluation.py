"""Evaluation metrics (Sec 5.3 - 5.5).

* :func:`edge_reproduction` -- percentage of ground-truth edges reproduced
  (the "accuracy" of the edge verification test, Table 2).
* :func:`map_metrics` -- length/edge-based precision, recall and F1 of a
  reconstructed map against ground truth.
* :func:`graph_similarity` -- graph-edit-distance similarity (Table 4).  The
  paper uses NetworkX ``optimize_graph_edit_distance``; exact GED is NP-hard,
  so two variants are offered: the NetworkX search (anytime, with a timeout)
  and a spatially anchored GED in which nodes are matched by location, which
  is exact for that matching and scales to city-sized graphs.
* :func:`eta_squared` -- ANOVA (type II, main effects) with eta^2 and partial
  eta^2 per factor (Sec 5.2, 5.4, Table 3).
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .geo import SegmentIndex, densify
from .roadmap import RoadMap


# --------------------------------------------------------------------------- #
# Edge reproduction / map metrics
# --------------------------------------------------------------------------- #
def _covered_fraction(segments: np.ndarray, *, other_points: np.ndarray | None = None,
                      other_segments: np.ndarray | None = None, theta: float,
                      step: float) -> np.ndarray:
    """For each segment, the fraction of its densified points within ``theta``
    of the other geometry (points or segments)."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    if len(segments) == 0:
        return np.zeros(0)
    pts, owner = [], []
    for i, s in enumerate(segments):
        d = densify(s, step)
        pts.append(d)
        owner.append(np.full(len(d), i))
    pts = np.vstack(pts)
    owner = np.concatenate(owner)
    if other_segments is not None and len(other_segments):
        _, dist = SegmentIndex(other_segments, sample_spacing=max(step, 1.0)).nearest(pts, theta)
        hit = np.isfinite(dist)
    elif other_points is not None and len(other_points):
        dist, _ = cKDTree(np.asarray(other_points)).query(pts, distance_upper_bound=theta)
        hit = np.isfinite(dist)
    else:
        hit = np.zeros(len(pts), dtype=bool)
    return np.bincount(owner, weights=hit, minlength=len(segments)) / np.bincount(owner, minlength=len(segments))


def edge_reproduction(truth_segments: np.ndarray, *, points: np.ndarray | None = None,
                      segments: np.ndarray | None = None, theta: float = 10.0,
                      min_fraction: float = 0.5, step: float = 2.0) -> np.ndarray:
    """Boolean mask: is each ground-truth edge reproduced?

    An edge counts as reproduced when at least ``min_fraction`` of it lies
    within ``theta`` of the reconstruction (the same majority-overlap rule as
    the agreement function, Algorithm 5).
    """
    frac = _covered_fraction(truth_segments, other_points=points, other_segments=segments,
                             theta=theta, step=step)
    return frac >= min_fraction


def map_metrics(truth: RoadMap, result: RoadMap, theta: float = 10.0, step: float = 2.0) -> dict:
    """Edge- and length-based precision / recall / F1 of ``result`` vs ``truth``."""
    ts, rs = truth.segments(), result.segments()
    if len(rs) == 0:
        return dict(edge_recall=0.0, edge_precision=0.0, length_recall=0.0,
                    length_precision=0.0, f1=0.0)
    tf = _covered_fraction(ts, other_segments=rs, theta=theta, step=step)
    rf = _covered_fraction(rs, other_segments=ts, theta=theta, step=step)
    tl = np.linalg.norm(ts[:, 1] - ts[:, 0], axis=1)
    rl = np.linalg.norm(rs[:, 1] - rs[:, 0], axis=1)
    lr = float((tf * tl).sum() / max(tl.sum(), 1e-9))
    lp = float((rf * rl).sum() / max(rl.sum(), 1e-9))
    return dict(edge_recall=float(np.mean(tf >= 0.5)), edge_precision=float(np.mean(rf >= 0.5)),
                length_recall=lr, length_precision=lp,
                f1=0.0 if lr + lp == 0 else 2 * lr * lp / (lr + lp))


# --------------------------------------------------------------------------- #
# Graph edit distance similarity (Sec 5.5)
# --------------------------------------------------------------------------- #
def _graph_size(g: nx.Graph) -> int:
    return g.number_of_nodes() + g.number_of_edges()


def anchored_graph_edit_distance(truth: RoadMap, result: RoadMap, tolerance: float = 10.0) -> int:
    """GED under a spatial node matching.

    Nodes of ``result`` are matched one-to-one to nodes of ``truth`` within
    ``tolerance`` metres (minimum total distance assignment).  With the
    matching fixed, the edit distance is the number of unmatched nodes on
    both sides plus the edges present in only one of the graphs.
    """
    tp, rp = truth.node_positions(), result.node_positions()
    tn, rn = list(tp), list(rp)
    match: dict = {}
    if tn and rn:
        ta = np.array([tp[n] for n in tn])
        ra = np.array([rp[n] for n in rn])
        pairs = cKDTree(ra).query_ball_tree(cKDTree(ta), tolerance)
        # solve the assignment per connected group of candidate pairs
        bip = nx.Graph()
        for i, js in enumerate(pairs):
            for j in js:
                bip.add_edge(("r", i), ("t", j))
        for comp in nx.connected_components(bip):
            ri = sorted(k[1] for k in comp if k[0] == "r")
            tj = sorted(k[1] for k in comp if k[0] == "t")
            cost = np.linalg.norm(ra[ri][:, None, :] - ta[tj][None, :, :], axis=2)
            cost = np.where(cost <= tolerance, cost, 1e9)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < 1e9:
                    match[rn[ri[r]]] = tn[tj[c]]
    unmatched_nodes = (len(rn) - len(match)) + (len(tn) - len(match))
    mapped = set()
    for u, v in result.graph.edges():
        if u in match and v in match:
            a, b = match[u], match[v]
            mapped.add((a, b) if str(a) <= str(b) else (b, a))
        else:
            mapped.add(("__r", u, v))
    truth_edges = {(u, v) if str(u) <= str(v) else (v, u) for u, v in truth.graph.edges()}
    edge_cost = len(mapped ^ truth_edges)
    return int(unmatched_nodes + edge_cost)


def networkx_graph_edit_distance(truth: RoadMap, result: RoadMap, tolerance: float = 10.0,
                                 timeout: float = 30.0) -> float:
    """Best GED found by ``networkx.optimize_graph_edit_distance`` in ``timeout`` s.

    Node substitution is free for nodes within ``tolerance`` metres and costs
    2 (= delete + insert) otherwise, so the search is spatially aware.
    Only practical for small graphs.
    """
    g1, g2 = nx.Graph(), nx.Graph()
    for src, dst in ((truth, g1), (result, g2)):
        for n, p in src.node_positions().items():
            dst.add_node(n, pos=p)
        dst.add_edges_from(src.graph.edges())

    def subst(a, b):
        return 0.0 if np.linalg.norm(a["pos"] - b["pos"]) <= tolerance else 2.0

    best = float(_graph_size(g1) + _graph_size(g2))
    start = time.time()
    for v in nx.optimize_graph_edit_distance(g1, g2, node_subst_cost=subst):
        best = min(best, v)
        if time.time() - start > timeout:
            break
    return best


def graph_similarity(truth: RoadMap, result: RoadMap, tolerance: float = 10.0,
                     method: str = "anchored", timeout: float = 30.0) -> float:
    """Similarity in [0, 1]: ``1 - GED / (|V1| + |E1| + |V2| + |E2|)``."""
    denom = _graph_size(truth.graph) + _graph_size(result.graph)
    if denom == 0:
        return 1.0
    if method == "anchored":
        ged = anchored_graph_edit_distance(truth, result, tolerance)
    elif method == "networkx":
        ged = networkx_graph_edit_distance(truth, result, tolerance, timeout)
    else:
        raise ValueError("method must be 'anchored' or 'networkx'")
    return float(1.0 - ged / denom)


# --------------------------------------------------------------------------- #
# ANOVA / eta-squared (Sec 5.2, 5.4)
# --------------------------------------------------------------------------- #
def eta_squared(df: pd.DataFrame, response: str, factors: list[str]) -> pd.DataFrame:
    """Type-II ANOVA with every factor categorical (main effects only).

    Returns the ANOVA table with ``eta_sq = SS_f / SS_total`` and
    ``partial_eta_sq = SS_f / (SS_f + SS_residual)``.  On the ProteusData
    Downtown Houston results this reproduces Table 3 of the paper exactly.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    names = {c: f"f{i}" for i, c in enumerate(factors)}
    data = df.rename(columns={**names, response: "y"})
    formula = "y ~ " + " + ".join(f"C({names[c]})" for c in factors)
    model = smf.ols(formula, data=data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    inv = {f"C({v})": k for k, v in names.items()}
    table.index = [inv.get(i, i) for i in table.index]
    ss_total = table["sum_sq"].sum()
    ss_res = table.loc["Residual", "sum_sq"]
    table["eta_sq"] = table["sum_sq"] / ss_total
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + ss_res)
    table.loc["Residual", ["eta_sq", "partial_eta_sq"]] = np.nan
    return table


def fit_anova_model(df: pd.DataFrame, response: str, factors: list[str]):
    """Fitted statsmodels OLS for ANOVA diagnostics (Figs. 9-10)."""
    import statsmodels.formula.api as smf

    names = {c: f"f{i}" for i, c in enumerate(factors)}
    data = df.rename(columns={**names, response: "y"})
    return smf.ols("y ~ " + " + ".join(f"C({names[c]})" for c in factors), data=data).fit()
