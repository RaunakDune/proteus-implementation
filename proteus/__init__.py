"""Proteus: a system for combining paths collected using crowdsensing.

Reference implementation of the algorithms in

    R. Sarbajna, C. F. Eick, J. Ni. "Proteus: A system for combining paths
    collected using crowdsensing." GeoInformatica 30:7 (2026).
    https://doi.org/10.1007/s10707-026-00567-1
"""

from .assignment import (AgentStart, deploy_poisson, deploy_quadrants, deploy_random,
                         distribute_agents)
from .combination import agree, agreement_matrix, combine_maps
from .evaluation import (edge_reproduction, eta_squared, graph_similarity, map_metrics)
from .geo import LocalProjection
from .intersections import (build_combined_map, graph_to_roadmap, graph_to_segments,
                            handle_intersections, intersect)
from .perturbation import PerturbationParams, apply_deletion, perturb_segments
from .pipeline import Proteus, ProteusConfig, ProteusResult
from .roadmap import RoadMap
from .simulation import AgentRules, GPSModel, Simulation
from .verification import VerifiedEdge, verify_traces

__version__ = "1.0.0"

__all__ = [
    "AgentRules", "AgentStart", "GPSModel", "LocalProjection", "PerturbationParams", "Proteus",
    "ProteusConfig", "ProteusResult", "RoadMap", "Simulation", "VerifiedEdge", "agree",
    "agreement_matrix", "apply_deletion", "build_combined_map", "combine_maps",
    "deploy_poisson", "deploy_quadrants", "deploy_random", "distribute_agents",
    "edge_reproduction", "eta_squared", "graph_similarity", "graph_to_roadmap",
    "graph_to_segments", "handle_intersections", "intersect", "map_metrics",
    "perturb_segments", "verify_traces",
]
