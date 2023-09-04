"""assignment cost calculation & matching methods."""

from __future__ import annotations

from abc import abstractmethod
from collections import namedtuple

import networkx as nx
import numpy as np

from sportslabkit import Tracklet
from sportslabkit.logger import logger
from sportslabkit.types.detection import Detection


EPS = 1e-7
# Define the named tuple outside of the function.
Node = namedtuple("Node", ["frame", "detection", "is_dummy"])


class BaseBatchMatchingFunction:
    """A base class for batch matching functions.

    A batch matching function takes a list of trackers and a list of detections
    and returns a list of matches.
    """

    def __call__(self, trackers: list[Tracklet], list_of_detections: list[list[Detection]]) -> list[int]:
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

    @abstractmethod
    def compute_cost_matrices(
        self, trackers: list[Tracklet], list_of_detections: list[list[Detection]]
    ) -> list[np.ndarray]:
        """Calculate the cost matrix between trackers and detections.

        Args:
            trackers: A list of trackers.
            list_of_detections: A list containing a list of detections for each frame.

        Returns:
            A list of 2D numpy arrays where the element at [i, j] in the kth array is the cost between tracker i and detection j in frame k.
        """
        pass

    @abstractmethod
    def _convert_cost_matrix_to_graph(
        self, cost_matrices: list[np.ndarray], no_detection_cost: float = 1e5
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], dict[int, Node], int, int]:
        """Transforms cost matrix into graph representation for min cost flow computation.

        Args:
            cost_matricies: A list of 2D numpy arrays where the element at [i, j] in the kth array is the cost between tracker i and detection j in frame k.

        Returns:
            A tuple containing arrays of start nodes, end nodes, capacities, unit costs, and supplies.
        """
        pass
