"""assignment cost calculation & matching methods."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from sportslabkit import Tracklet
from sportslabkit.types.detection import Detection


EPS = 1e-7
from typing import List

import networkx as nx

from sportslabkit import Tracklet
from sportslabkit.logger import logger
from sportslabkit.types.detection import Detection


class BaseBatchMatchingFunction:
    """A base class for batch matching functions.

    A batch matching function takes a list of trackers and a list of detections
    and returns a list of matches.
    """

    def __call__(self, trackers: List[Tracklet], list_of_detections: List[List[Detection]]) -> List[List[int]]:
        """Calculate the matching cost between trackers and detections and performs matching.

        Args:
            trackers: A list of trackers.
            list_of_detections: A list containing a list of detections for each frame.

        Returns:
            A list for each tracklet containing the indices of the matched detections for each frame. For example, if `matches[0][2] = 5`, it means the tracker with index 0 was matched to the detection with index 5 in thethird frame.
        """

        cost_matricies = self.compute_cost_matrices(trackers, list_of_detections)

        n_trackers = len(trackers)
        n_frames = len(list_of_detections)
        n_detections_average = np.mean([len(dets) for dets in list_of_detections])
        logger.debug(f"(n_trackers, n_detections(ave), n_frames)=({n_trackers}, {n_detections_average}, {n_frames})")
        G = self._convert_cost_matrix_to_graph(cost_matricies)

        flow_path = nx.min_cost_flow(G)

        valid_path = []
        for node in flow_path:
            for neighbor, flow in flow_path[node].items():
                if flow > 0:
                    next_node = G.nodes[neighbor]
                    if next_node["demand"] != 0:
                        continue
                    if next_node["is_dummy"]:
                        valid_path.append(-1)
                    else:
                        valid_path.append(next_node["detection"])
        return valid_path
        # (
        #     start_nodes,
        #     end_nodes,
        #     capacities,
        #     unit_costs,
        #     supplies,
        #     node_to_detection,
        #     source_node,
        #     sink_node,

        # smcf = min_cost_flow.SimpleMinCostFlow()

        # # Add arcs, capacities and costs in bulk using numpy.
        # all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities, unit_costs)

        # # Add supply for each nodes.
        # smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

        # if smcf.solve() == smcf.OPTIMAL:
        #     _flow_paths = self.build_flow_paths(source_node, sink_node, smcf)
        #     flow_paths = []
        #     for _path in _flow_paths:
        #         flow_paths.append([node_to_detection[node] for node in _path])
        # else:
        #     logger.debug("There was an issue with the min cost flow input.")
        # return flow_paths

    # def build_flow_paths(self, source_node, sink_node, smcf):
    #     flow_arcs = {}
    #     for arc in range(smcf.num_arcs()):
    #         if smcf.flow(arc) > 0:
    #             flow_arcs[smcf.tail(arc)] = smcf.head(arc)

    #     paths = []
    #     while source_node in flow_arcs:
    #         path = [source_node]
    #         while path[-1] != sink_node:
    #             path.append(flow_arcs[path[-1]])
    #         # Exclude the source and sink nodes from the path.
    #         paths.append(path[1:-1])
    #         # Remove the used path
    #         for node in path[:-1]:
    #             del flow_arcs[node]
    #     return paths

    @abstractmethod
    def compute_cost_matrices(
        self, trackers: List[Tracklet], list_of_detections: List[List[Detection]]
    ) -> List[np.ndarray]:
        """Calculate the cost matrix between trackers and detections.

        Args:
            trackers: A list of trackers.
            list_of_detections: A list containing a list of detections for each frame.

        Returns:
            A list of 2D numpy arrays where the element at [i, j] in the kth array is the cost between tracker i and detection j in frame k.
        """
        pass

    @abstractmethod
    def _convert_cost_matrix_to_graph(self, cost_matricies: np.ndarray) -> tuple:
        """Transforms cost matrix into graph representation for min cost flow computation.

        Args:
            cost_matricies: A list of 2D numpy arrays where the element at [i, j] in the kth array is the cost between tracker i and detection j in frame k.

        Returns:
            A tuple containing arrays of start nodes, end nodes, capacities, unit costs, and supplies.
        """
        pass
