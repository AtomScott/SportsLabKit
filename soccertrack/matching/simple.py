from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from typing import Any, Callable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from soccertrack import Tracklet
from soccertrack.logger import logger
from soccertrack.matching.base import BaseMatchingFunction
from soccertrack.matching.base_batch import BaseBatchMatchingFunction
from soccertrack.metrics import BaseCostMatrixMetric, CosineCMM, IoUCMM
from soccertrack.types.detection import Detection

# Define the named tuple outside of the function.
Node = namedtuple("Node", ["frame", "detection", "is_dummy"])


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

    def compute_cost_matrix(
        self, trackers: Sequence[Tracklet], detections: Sequence[Detection]
    ) -> np.ndarray:
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


from typing import List

import numpy as np

from soccertrack import Tracklet
from soccertrack.types.detection import Detection


class SimpleBatchMatchingFunction(BaseBatchMatchingFunction):
    """A batch matching function that uses a simple distance metric.

    This class is a simple implementation of batch matching function where the cost is based on the Euclidean distance between the trackers and detections.
    """

    def compute_cost_matrices(
        self, trackers: List[Tracklet], list_of_detections: List[List[Detection]]
    ) -> List[np.ndarray]:
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
                        cost_matrix[i, j] = np.linalg.norm(
                            np.array(tracker.box) - np.array(detection.box)
                        )
            else:  # Other frames
                prev_detections = list_of_detections[k - 1]
                cost_matrix = np.zeros((len(prev_detections), num_detections))
                for j, detection in enumerate(detections):
                    for i in range(
                        min(len(prev_detections), j + 1)
                    ):  # j-1 could be negative, so we make sure it's at least 0
                        cost_matrix[i, j] = np.linalg.norm(
                            np.array(prev_detections[i].box) - np.array(detection.box)
                        )

            cost_matrices.append(cost_matrix)

        return cost_matrices

    def _convert_cost_matrix_to_graph(
        self, cost_matrices: List[np.ndarray], no_detection_cost: float = 1e5
    ) -> Tuple[
        List[int], List[int], List[int], List[int], List[int], Dict[int, Node], int, int
    ]:
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
        frame_to_nodes = defaultdict(list)  # keep track of the nodes for each frame

        num_frames = len(cost_matrices)
        source_node = 0
        sink_node = 10**9
        G.add_node(source_node, demand=-1, frame=-1)
        G.add_node(sink_node, demand=1, frame=num_frames)

        curr_node = 1
        for frame in range(num_frames):
            num_detections_curr_frame = (
                cost_matrices[frame].shape[1] + 1
            )  # +1 for dummy node
            num_detections_prev_frame = len(frame_to_nodes[frame - 1])

            for detection_curr in range(num_detections_curr_frame):
                is_dummy_node = detection_curr + 1 == num_detections_curr_frame

                if frame == 0:  # For the first frame, source_node -> node
                    cost = (
                        no_detection_cost
                        if is_dummy_node
                        else cost_matrices[0][0][detection_curr]
                    )
                    G.add_node(
                        curr_node,
                        demand=0,
                        frame=frame,
                        detection=detection_curr,
                        is_dummy=is_dummy_node,
                    )
                    G.add_edge(source_node, curr_node, capacity=1, weight=cost)
                    if curr_node not in frame_to_nodes[frame]:
                        frame_to_nodes[frame].append(curr_node)
                else:
                    for detection_prev in range(num_detections_prev_frame):
                        is_dummy_node_prev = (
                            detection_prev + 1
                        ) == num_detections_prev_frame

                        no_detection = (
                            is_dummy_node
                            or is_dummy_node_prev
                            or detection_prev >= num_detections_prev_frame
                            or detection_curr >= num_detections_curr_frame
                        )
                        # if not no_detection:
                        # print(cost_matrices[frame][detection_prev], detection_prev)
                        cost = (
                            no_detection_cost
                            if no_detection
                            else cost_matrices[frame][detection_prev][detection_curr]
                        )

                        prev_node = frame_to_nodes[frame - 1][detection_prev]
                        # curr_node = detection_curr + frame * num_detections_curr_frame
                        G.add_node(
                            curr_node,
                            demand=0,
                            frame=frame,
                            detection=detection_curr,
                            is_dummy=is_dummy_node,
                        )
                        G.add_edge(prev_node, curr_node, capacity=1, weight=cost)
                        if curr_node not in frame_to_nodes[frame]:
                            frame_to_nodes[frame].append(curr_node)
                curr_node += 1

            if frame == num_frames - 1:
                for node in frame_to_nodes[frame]:
                    G.add_edge(node, sink_node, capacity=1, weight=0)
        return G
        # num_frames = len(cost_matrices)

        # source_node = 0
        # sink_node = 10**9
        # start_nodes, end_nodes, capacities, unit_costs, supplies = [], [], [], [], []
        # node_to_detection = {source_node: Node(-1, -1, False), sink_node: Node(-1, -1, False)}

        # def add_arc(start_node, end_node, capacity, cost):
        #     """Helper function to add an arc to the graph."""
        #     start_nodes.append(int(start_node))
        #     end_nodes.append(int(end_node))
        #     capacities.append(int(capacity))
        #     unit_costs.append(int(cost))
        #     logger.debug(f"Adding arc from {start_node} to {end_node} with capacity {capacity} and cost {cost}")

        # for frame in range(num_frames):
        #     num_detections_curr_frame = cost_matrices[frame].shape[1] + 1  # +1 for dummy node

        #     logger.debug(f"Adding {num_detections_curr_frame-1} + 1 (dummy) nodes for frame {frame}")

        #     for detection_curr in range(num_detections_curr_frame):
        #         curr_node = len(start_nodes) + 1
        #         is_dummy_node = detection_curr + 1 == num_detections_curr_frame
        #         node_to_detection[curr_node] = Node(frame, detection_curr, is_dummy_node)
        #         logger.debug(f"frame: {frame}, node_num{curr_node}, is_dummy_node: {is_dummy_node} ")

        #         if frame == 0:  # For the first frame, source_node -> node
        #             cost = no_detection_cost if is_dummy_node else cost_matrices[0][0][detection_curr]
        #             add_arc(source_node, curr_node, 1, cost)
        #         else:
        #             num_detections_prev_frame = cost_matrices[frame - 1].shape[1] + 1
        #             for detection_prev in range(num_detections_prev_frame):
        #                 is_dummy_node_prev = (detection_prev + 1) == num_detections_prev_frame

        #                 no_detection = (
        #                     is_dummy_node
        #                     or is_dummy_node_prev
        #                     or detection_prev >= num_detections_prev_frame
        #                     or detection_curr >= num_detections_curr_frame
        #                 )
        #                 cost = (
        #                     no_detection_cost if no_detection else cost_matrices[frame][detection_prev][detection_curr]
        #                 )

        #                 prev_node = detection_prev + (frame - 1) * num_detections_prev_frame
        #                 curr_node = detection_curr + frame * num_detections_curr_frame
        #                 add_arc(prev_node, curr_node, 1, cost)

        #     if frame == num_frames - 1:
        #         for detection in range(num_detections_curr_frame):
        #             curr_node = detection + frame * num_detections_curr_frame
        #             add_arc(curr_node, sink_node, 1, 0)
        #             logger.debug(f"Add arc from node {curr_node} to sink node({sink_node})")

        # supplies = [0] * source_node
        # supplies.extend([1, -1])

        # return start_nodes, end_nodes, capacities, unit_costs, supplies, node_to_detection, source_node, sink_node


# %matplotlib inline
# import matplotlib.pyplot as plt
# def visualize_graph(G):
#     """
#     Function to visualize a graph with 'frame' attributes
#     Args:
#     G: networkx.DiGraph() object
#     """
#     # Ensure that the graph is a DiGraph
#     if not isinstance(G, nx.DiGraph):
#         raise ValueError("G must be a networkx DiGraph")

#     # Get the 'frame' attribute for each node and sort nodes by this attribute
#     frame_values = nx.get_node_attributes(G, 'frame')
#     sorted_nodes = sorted(G.nodes(), key=lambda x: frame_values[x])

#     # Create a layout for the graph, placing nodes with lower 'frame' values to the left
#     pos = {node: (frame_values[node], i) for i, node in enumerate(sorted_nodes)}

#     # Increase the figure size
#     plt.figure(figsize=(20, 15))

#     # Draw the graph with smaller nodes and thinner edges
#     nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue',
#             font_weight='bold', width=0.2, alpha=0.7)

#     plt.show()

# visualize_graph(G)
