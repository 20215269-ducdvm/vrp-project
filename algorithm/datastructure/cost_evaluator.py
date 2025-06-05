import sys

import numpy as np
import heapq
from collections import defaultdict
from itertools import cycle
import math
from math import floor
from typing import Any

from helpers.helpers import compute_euclidean_distance
from datastructure.matrices.distance_matrix import DistanceMatrix
from .matrices.duration_matrix import create_duration_matrix
from .metrics import BaseMetric

from .node import Node
from .edge import Edge
from .penalty_manager import PenaltyWeights
from .route import Route
from .segment.DistanceSegment import Distance
from .segment.DurationSegment import Duration
from .segment.LoadSegment import Load
from .vrp_solution import VRPSolution


class Cost(BaseMetric['Cost']):
    pass


class MaxHeapWithUpdate:
    def __init__(self, elements: list[Edge]):
        self.heap = elements
        heapq.heapify(self.heap)

    def get_max_element(self):
        return heapq.heappop(self.heap)

    def insert_element(self, element: Edge):
        heapq.heappush(self.heap, element)

    def get_sorted_list(self):
        return sorted([elem for elem in self.heap])


class CostEvaluator:
    def __init__(self, nodes: list[Node], capacity: int, run_parameters: dict[str, Any]):
        self._penalization_enabled: bool = False
        self._edge_penalties: dict[Edge, int] = defaultdict(int)
        self._edge_ranking: MaxHeapWithUpdate = None
        # size of the neighborhood during search process
        self.neighborhood_size = run_parameters['neighborhood_size']
        self._capacity = capacity

        # Create the distance matrix
        self._distance_matrix = DistanceMatrix(nodes)
        self._nodes = nodes
        self.node_to_idx = self._distance_matrix.node_to_idx
        self.idx_to_node = self._distance_matrix.idx_to_node

        # Initialize penalized costs as a copy of the distance matrix
        self._costs = self._distance_matrix.get_matrix()
        self._penalized_costs = self._costs.copy()

        # Get neighborhood for each node
        self._neighborhood = self._compute_neighborhood(nodes)

        # Calculate baseline cost
        neighborhood_costs = []
        for node in nodes:
            if not node.is_depot:
                for other in self._neighborhood[node]:
                    idx1 = self.node_to_idx[node.node_id]
                    idx2 = self.node_to_idx[other.node_id]
                    neighborhood_costs.append(self._costs[idx1, idx2])

        self._baseline_cost = int(sum(neighborhood_costs) / (self.neighborhood_size * len(nodes)))

        self._penalization_criterium_options = cycle(["width", "length", "width_length"])
        self._penalization_criterium = next(self._penalization_criterium_options)

        self._infeasible_penalty_weights = None
        self._duration_matrix = None

    def set_infeasible_penalty_weights(self, load: float, time_warp: float):
        """
        Set the penalized cost evaluator.
        """
        self._infeasible_penalty_weights = PenaltyWeights(load, time_warp)

    def set_duration_matrix(self, calculate_method="euclidean"):
        """
        Set the duration matrix for the cost evaluator.
        """
        self._duration_matrix = create_duration_matrix(self._nodes, calculate_method)

    def _compute_infeasible_solution_cost(self, solution: VRPSolution) -> float:
        """
        Calculate the penalized costs of the solution based on the penalty weights.
        """
        if self._infeasible_penalty_weights is None:
            raise ValueError("Penalized cost evaluator not set.")

        distance, load, tw = solution.get_solution_penalties()

        # Calculate the penalized costs based on the parameters
        penalized_cost = (distance
                          + self._infeasible_penalty_weights.load_penalty * load)

        if not hasattr(sys, "VRP_NO_TIME_WINDOWS"):
            penalized_cost += self._infeasible_penalty_weights.time_window_penalty * tw

        return penalized_cost

    def get_route_cost(self, distance: int, load: int, tw: int) -> float:
        if self._infeasible_penalty_weights is None:
            raise ValueError("Penalized cost evaluator not set.")

        penalized_cost = (int(distance)
                          + self._infeasible_penalty_weights.load_penalty * load)

        if not hasattr(sys, "VRP_NO_TIME_WINDOWS"):
            penalized_cost += self._infeasible_penalty_weights.time_window_penalty * tw

        return penalized_cost

    def get_neighborhood(self, node: Node) -> list[Node]:
        return self._neighborhood[node]

    def _compute_neighborhood(self, nodes: list[Node]) -> dict[Node, list[Node]]:
        neighborhood = {
            node: self._get_nearest_neighbors(node, nodes)
            for node in nodes
            if not node.is_depot
        }
        return neighborhood

    def _get_nearest_neighbors(self, node: Node, nodes: list[Node]) -> list[Node]:
        idx1 = self.node_to_idx[node.node_id]

        # Get distances to all nodes
        distances = [(self._costs[idx1, self.node_to_idx[n.node_id]], n)
                     for n in nodes if not n.is_depot and n != node]

        # Sort by distance
        distances.sort()

        # Take the nearest neighbors
        return [n for _, n in distances[:self.neighborhood_size]]

    def is_feasible(self, capacity: int) -> bool:
        return capacity <= self._capacity

    def determine_edge_badness(self, routes: list[Route]):
        edges_in_solution: list[Edge] = []

        criterium_functions = {
            "length": self._compute_edge_length_value,
            "width": self._compute_edge_width_value,
            "width_length": self._compute_edge_width_length_value
        }
        # Get the computation function based on the current penalization criterium
        compute_edge_value = criterium_functions[self._penalization_criterium]

        for route in routes:
            center_x, center_y = (None, None)
            if self._penalization_criterium in {"width", "width_length"}:
                center_x, center_y = self._compute_route_center(route.nodes)

            for edge in route.edges:
                # Compute the value for the edge
                edge.value = compute_edge_value(edge, center_x, center_y, route)
                edge.value /= (1 + self._edge_penalties[edge])
                edges_in_solution.append(edge)

        # Update edge ranking
        self._edge_ranking = MaxHeapWithUpdate(edges_in_solution)

        # Rotate to next penalization criterium
        self._penalization_criterium = next(self._penalization_criterium_options)

    def _compute_edge_length_value(self, edge: Edge, *args) -> float:
        idx1 = self.node_to_idx[edge.nodes[0].node_id]
        idx2 = self.node_to_idx[edge.nodes[1].node_id]
        return self._costs[idx1, idx2]

    def _compute_edge_width_value(self, edge: Edge, center_x: float, center_y: float, route: Route) -> float:
        return self._compute_edge_width(edge, center_x, center_y, route.depot)

    def _compute_edge_width_length_value(self, edge: Edge, center_x: float, center_y: float, route: Route) -> float:
        width_value = self._compute_edge_width(edge, center_x, center_y, route.depot)

        idx1 = self.node_to_idx[edge.nodes[0].node_id]
        idx2 = self.node_to_idx[edge.nodes[1].node_id]
        length_value = self._costs[idx1, idx2]

        return width_value + length_value

    def enable_penalization(self):
        self._penalization_enabled = True

    def disable_penalization(self):
        self._penalization_enabled = False

    def get_distance(self, node1: Node, node2: Node) -> float:
        idx1 = self.node_to_idx[node1.node_id]
        idx2 = self.node_to_idx[node2.node_id]

        # Check if the distance is stored (not infinity)
        if np.isinf(self._costs[idx1, idx2]):
            # Calculate on the fly if not stored
            distance = compute_euclidean_distance(node1, node2)
            if not self._penalization_enabled:
                return distance
            else:
                return floor(distance + 0.1 * self._baseline_cost * self._edge_penalties[Edge(node1, node2)])

        # Use stored distance
        if not self._penalization_enabled:
            return self._costs[idx1, idx2]
        else:
            return self._penalized_costs[idx1, idx2]

    def get_and_penalize_worst_edge(self) -> Edge:
        worst_edge = self._edge_ranking.get_max_element()
        self._edge_penalties[worst_edge] += 1

        # update costs
        idx1 = self.node_to_idx[worst_edge.nodes[0].node_id]
        idx2 = self.node_to_idx[worst_edge.nodes[1].node_id]

        penalization_costs = round(
            self._costs[idx1, idx2]
            + 0.1 * self._baseline_cost * self._edge_penalties[worst_edge]
        )
        self._penalized_costs[idx1, idx2] = penalization_costs
        self._penalized_costs[idx2, idx1] = penalization_costs

        # update (reduce) 'badness' of the just penalized edge
        worst_edge.value = (
                self._costs[idx1, idx2] /
                (1 + self._edge_penalties[worst_edge])
        )
        self._edge_ranking.insert_element(worst_edge)

        return worst_edge

    def penalize(self, edge: Edge) -> None:
        self._edge_penalties[edge] += 1

    def get_solution_costs(self, solution: VRPSolution, ignore_penalties: bool = False,
                           ) -> float:
        solution_costs: int = 0

        if self._infeasible_penalty_weights is None:
            raise ValueError("Infeasible cost evaluator not set.")

        if not solution.is_feasible():
            # If the solution is infeasible, use the penalized costs
            return self._compute_infeasible_solution_cost(solution)

        # If the solution is feasible, calculate the costs based on the routes
        for route in solution.routes:
            if route.size > 0:
                for idx in range(len(route.all_nodes) - 1):
                    edge_node1 = route.all_nodes[idx]
                    edge_node2 = route.all_nodes[idx + 1]
                    if ignore_penalties:
                        idx1 = self.node_to_idx[edge_node1.node_id]
                        idx2 = self.node_to_idx[edge_node2.node_id]
                        solution_costs += self._costs[idx1, idx2]
                    else:
                        solution_costs += self.get_distance(edge_node1, edge_node2)

        return Cost(solution_costs)

    @staticmethod
    def _compute_edge_width(
            edge: Edge,
            route_center_x: float,
            route_center_y: float,
            depot: Node
    ) -> float:
        node1 = edge.get_first_node()
        node2 = edge.get_second_node()

        distance_depot_center = (
            math.sqrt(
                math.pow(depot.x_coordinate - route_center_x, 2) +
                math.pow(depot.y_coordinate - route_center_y, 2)
            )
        )

        distance_node1 = (
                (route_center_y - depot.y_coordinate) * node1.x_coordinate
                - (route_center_x - depot.x_coordinate) * node1.y_coordinate
                + (route_center_x * depot.y_coordinate) - (route_center_y * depot.x_coordinate)
        )
        distance_node1 = 0 if distance_depot_center == 0 else distance_node1 / distance_depot_center

        distance_node2 = (
                (route_center_y - depot.y_coordinate) * node2.x_coordinate
                - (route_center_x - depot.x_coordinate) * node2.y_coordinate
                + (route_center_x * depot.y_coordinate) - (route_center_y * depot.x_coordinate)
        )
        distance_node2 = 0 if distance_depot_center == 0 else distance_node2 / distance_depot_center

        return abs(distance_node1 - distance_node2)

    @staticmethod
    def _compute_route_center(nodes: list[Node]) -> tuple[float, float]:
        mean_x = sum(node.x_coordinate for node in nodes) / len(nodes)
        mean_y = sum(node.y_coordinate for node in nodes) / len(nodes)

        return mean_x, mean_y
