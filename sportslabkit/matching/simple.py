from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Sequence

import networkx as nx
import numpy as np

from sportslabkit import Tracklet
from sportslabkit.matching.base import BaseMatchingFunction
from sportslabkit.matching.base_batch import BaseBatchMatchingFunction, Node
from sportslabkit.metrics import BaseCostMatrixMetric, IoUCMM
from sportslabkit.types.detection import Detection


class SimpleMatchingFunction(BaseMatchingFunction):
    """A matching function that uses a single metric.

    Args:
        metric: A metric. Defaults to `IoUCMM`.
        gate: The gate of the metric, i.e. if the metric is larger than
            this value, the cost will be set to infinity. Defaults to `np.inf`.

    Note:
        To implement your own matching function, you can inherit from `BaseMatchingFunction`
        and override the :meth:`compute_cost_matrix` method.
    """

    def __init__(
        self,
        metric: BaseCostMatrixMetric = IoUCMM(),
        gate: float = np.inf,
    ) -> None:
        if not isinstance(metric, BaseCostMatrixMetric):
            raise TypeError("metric should be a BaseCostMatrixMetric")
        self.metric = metric
        self.gate = gate

    def compute_cost_matrix(self, trackers: Sequence[Tracklet], detections: Sequence[Detection]) -> np.ndarray:
        """Calculate the matching cost between trackers and detections.

        Args:
            trackers: A list of trackers.
            detections: A list of detections.

        returns:
            A 2D numpy array of matching costs between trackers and detections.
        """
        if len(trackers) == 0 or len(detections) == 0:
            return np.array([])

        cost_matrix = self.metric(trackers, detections)
        cost_matrix = cost_matrix
        cost_matrix[cost_matrix > self.gate] = np.inf
        return cost_matrix


class SimpleBatchMatchingFunction(BaseBatchMatchingFunction):
    """A batch matching function that uses a simple distance metric.

    This class is a simple implementation of batch matching function where the cost is based on the Euclidean distance between the trackers and detections.
    """

    def compute_cost_matrices(
        self, trackers: list[Tracklet], list_of_detections: list[list[Detection]]
    ) -> list[np.ndarray]:
        """Calculate the cost matrix between trackers and detections.

        Args:
            trackers: A list of trackers.
            list_of_detections: A list containing a list of detections for each frame.

        Returns:
            A list of 2D numpy arrays where the element at [i, j] in the kth array is the cost between tracker i and detection j in frame 0, detection j-1 and detection j otherwise.
        """

        cost_matrices = []

        for k, detections in enumerate(list_of_detections):
            num_detections = len(detections)
            if k == 0:  # First frame
                cost_matrix = np.zeros((len(trackers), num_detections))
                for i, tracker in enumerate(trackers):
                    for j, detection in enumerate(detections):
                        cost_matrix[i, j] = np.linalg.norm(np.array(tracker.box) - np.array(detection.box))
            else:  # Other frames
                prev_detections = list_of_detections[k - 1]
                cost_matrix = np.zeros((len(prev_detections), num_detections))
                for j, detection in enumerate(detections):
                    for i in range(
                        min(len(prev_detections), j + 1)
                    ):  # j-1 could be negative, so we make sure it's at least 0
                        cost_matrix[i, j] = np.linalg.norm(np.array(prev_detections[i].box) - np.array(detection.box))

            cost_matrices.append(cost_matrix)

        return cost_matrices

    def _convert_cost_matrix_to_graph(
        self, cost_matrices: list[np.ndarray], no_detection_cost: float = 1e5
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], dict[int, Node], int, int]:
        """
        Converts cost matrix to graph representation for optimization.

        Args:
            cost_matrices: List of Numpy arrays representing the cost matrices.
            no_detection_cost: Cost to be used when there is no detection.

        Returns:
            start_nodes: List of start nodes.
            end_nodes: List of end nodes.
            capacities: List of the capacities of the arcs.
            unit_costs: List of the unit costs of the arcs.
            supplies: List of the supplies for the nodes.
            node_to_detection: Dictionary mapping node to a Node namedtuple.
            source_node: Source node index.
            sink_node: Sink node index.
        """
        G = nx.DiGraph()
        frame_to_nodes: DefaultDict[int, list[int]] = defaultdict(list)  # keep track of the nodes for each frame

        num_frames = len(cost_matrices)
        source_node = 0
        sink_node = 10**9
        G.add_node(source_node, demand=-1, frame=-1)
        G.add_node(sink_node, demand=1, frame=num_frames)

        curr_node = 1
        for frame in range(num_frames):
            num_detections_curr_frame = cost_matrices[frame].shape[1] + 1  # +1 for dummy node
            num_detections_prev_frame = len(frame_to_nodes[frame - 1])

            for detection_curr in range(num_detections_curr_frame):
                is_dummy_node = detection_curr + 1 == num_detections_curr_frame

                if frame == 0:  # For the first frame, source_node -> node
                    cost = no_detection_cost if is_dummy_node else cost_matrices[0][0][detection_curr]
                    G.add_node(curr_node, demand=0, frame=frame, detection=detection_curr, is_dummy=is_dummy_node)
                    G.add_edge(source_node, curr_node, capacity=1, weight=cost)
                    if curr_node not in frame_to_nodes[frame]:
                        frame_to_nodes[frame].append(curr_node)
                else:
                    for detection_prev in range(num_detections_prev_frame):
                        is_dummy_node_prev = (detection_prev + 1) == num_detections_prev_frame

                        no_detection = (
                            is_dummy_node
                            or is_dummy_node_prev
                            or detection_prev >= num_detections_prev_frame
                            or detection_curr >= num_detections_curr_frame
                        )
                        # if not no_detection:
                        # print(cost_matrices[frame][detection_prev], detection_prev)
                        cost = (
                            no_detection_cost if no_detection else cost_matrices[frame][detection_prev][detection_curr]
                        )

                        prev_node = frame_to_nodes[frame - 1][detection_prev]
                        # curr_node = detection_curr + frame * num_detections_curr_frame
                        G.add_node(curr_node, demand=0, frame=frame, detection=detection_curr, is_dummy=is_dummy_node)
                        G.add_edge(prev_node, curr_node, capacity=1, weight=cost)
                        if curr_node not in frame_to_nodes[frame]:
                            frame_to_nodes[frame].append(curr_node)
                curr_node += 1

            if frame == num_frames - 1:
                for node in frame_to_nodes[frame]:
                    G.add_edge(node, sink_node, capacity=1, weight=0)
        return G
