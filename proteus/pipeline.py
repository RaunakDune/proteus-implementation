"""End-to-end Proteus pipeline (Fig. 1).

    traces --(Alg 2: point-wise verification)--> verified edges
           --(combine)--> combined map M
           --(Alg 3/4: intersection detection + handling)--> corrected map M_C
           --(Alg 5/6: agreement + map combination vs. pre-disaster map)--> final map
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .combination import CombinationResult, combine_maps
from .intersections import (IntersectionResult, build_combined_map, graph_to_roadmap,
                            graph_to_segments, handle_intersections,
                            verified_edges_to_segments)
from .roadmap import RoadMap
from .verification import VerificationResult, verify_traces


@dataclass
class ProteusConfig:
    # Algorithm 2
    threshold: float = 10.0            # theta: max tolerated GPS noise (m)
    min_support: int = 3               # n_p >= 3
    count_mode: str = "volunteers"     # "points" | "traces" | "volunteers"
    map_match: bool = True             # discard points farther than theta from M
    snap_to_map: bool = False
    # Combined map + Algorithms 3/4
    graph_spacing: float = 5.0         # resampling of verified edges (m)
    merge_radius: float | None = None  # default: threshold
    smooth_window: float = 10.0        # moving-average window for verified edges (m)
    heading_tolerance: float = 45.0    # deg, direction test when inserting tracks
    handle_intersections: bool = True
    intersection_radius: float = 15.0  # the paper's intersection threshold distance
    min_cycle_edges: int = 3
    # Algorithms 5/6
    angle_tolerance: float = 15.0      # omega (deg)
    segment_length: float = 20.0       # l: length of verified edges and of M's sub-edges (m)
    subdivision_length: object = None  # None -> segment_length; or "smallest" / "largest" / float
    min_subdivision_length: float = 10.0
    min_agreement: int = 2             # c >= 2
    segment_source: str = "verified"   # E for Alg 6: "verified" (bag of verified edges) or "combined" (M_C)
    agreement_reference: str = "verified"  # "verified" (Alg 6 line 6) or "map" (Alg 5 signature)
    min_component_edges: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProteusResult:
    config: ProteusConfig
    verification: VerificationResult
    combined_graph: nx.Graph                  # M (before intersection handling)
    intersections: IntersectionResult         # M_C
    trace_map: RoadMap                        # M_C as a lon/lat road map
    verified_segments: np.ndarray             # E fed to Algorithm 6 (metres)
    combination: CombinationResult | None
    final_map: RoadMap

    def summary(self) -> dict:
        v = self.verification
        out = {
            "points": int(len(v.points)),
            "points_map_matched": int(v.points["map_ok"].sum()) if len(v.points) else 0,
            "points_verified": int(v.points["verified"].sum()) if len(v.points) else 0,
            "verified_edges": len(v.edges),
            "combined_nodes": self.combined_graph.number_of_nodes(),
            "combined_edges": self.combined_graph.number_of_edges(),
            "cycles_collapsed": self.intersections.cycles_collapsed,
            "intersections_found": len(self.intersections.intersections),
            "segments_to_combination": int(len(self.verified_segments)),
            "final_edges": self.final_map.graph.number_of_edges(),
            "final_length_m": round(self.final_map.total_length(), 1),
        }
        if self.combination is not None:
            out["subdivision_length_m"] = round(self.combination.subdivision_length, 2)
            out["subedges_total"] = int(len(self.combination.kept))
            out["subedges_kept"] = int(self.combination.kept.sum())
        return out


class Proteus:
    """The Proteus map-making system.

    >>> proteus = Proteus(pre_disaster_map, ProteusConfig(threshold=10))
    >>> result = proteus.run(traces_df)
    >>> result.final_map.to_edge_csv("post_disaster_map.csv")
    """

    def __init__(self, pre_map: RoadMap | None = None, config: ProteusConfig | None = None,
                 projection=None):
        self.pre_map = pre_map
        self.config = config or ProteusConfig()
        self.proj = projection or (pre_map.proj if pre_map is not None else None)

    def run(self, traces: pd.DataFrame) -> ProteusResult:
        c = self.config
        if self.proj is None:
            from .geo import LocalProjection
            self.proj = LocalProjection.around(traces["lon"], traces["lat"])

        # Algorithm 2
        ver = verify_traces(traces, c.threshold, road_map=self.pre_map,
                            min_support=c.min_support, count_mode=c.count_mode,
                            map_match=c.map_match, snap_to_map=c.snap_to_map,
                            projection=self.proj)
        # Combine verified edges into one map
        merge_radius = c.merge_radius if c.merge_radius is not None else c.threshold
        combined = build_combined_map(ver.edges, spacing=c.graph_spacing,
                                      merge_radius=merge_radius,
                                      heading_tolerance=c.heading_tolerance,
                                      smooth_window=c.smooth_window)
        # Algorithms 3 + 4
        if c.handle_intersections:
            inter = handle_intersections(combined, c.intersection_radius,
                                         min_cycle_edges=c.min_cycle_edges)
        else:
            inter = IntersectionResult(graph=combined.copy())
        trace_map = graph_to_roadmap(inter.graph, self.proj)
        if c.segment_source == "verified":
            segs = verified_edges_to_segments(ver.edges, c.segment_length,
                                              spacing=c.graph_spacing,
                                              smooth_window=c.smooth_window)
        else:
            segs = graph_to_segments(inter.graph, c.segment_length)

        # Algorithms 5 + 6
        comb = None
        if self.pre_map is not None:
            sub_len = c.subdivision_length if c.subdivision_length is not None else c.segment_length
            comb = combine_maps(self.pre_map, segs, c.threshold, c.angle_tolerance,
                                subdivision_length=sub_len,
                                min_subdivision_length=c.min_subdivision_length,
                                min_agreement=c.min_agreement,
                                min_component_edges=c.min_component_edges,
                                reference=c.agreement_reference)
            final = comb.map
        else:
            final = trace_map
        return ProteusResult(config=c, verification=ver, combined_graph=combined,
                             intersections=inter, trace_map=trace_map,
                             verified_segments=segs, combination=comb, final_map=final)
