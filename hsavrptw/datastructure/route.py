import math
from typing import List

from datastructure.segment.DurationSegment import DurationSegment
from .edge import Edge
from .node import Node, NodeWithTW


class Route:
    volume: int

    def __init__(self, nodes: List[Node], route_index: int):
        # Initialize the route with the depot as both the first and last node.
        assert nodes[0].is_depot, 'First node of a route has to be a depot.'
        assert nodes[-1].is_depot, 'Last node of a route has to be a depot.'
        assert nodes[0] == nodes[-1], 'Start and return depot has to be the same'

        self.route_index = route_index
        self.depot: Node = nodes[0]
        self._nodes: list = nodes.copy()

        self.size = len(nodes) - 2  # Number of customers (not including depot)

        self.volume = sum(node.demand for node in self._nodes)  # Sum of demand of all customers of the route

        self.validate()

    def __repr__(self):
        return '-'.join([str(node.node_id) for node in self._nodes])

    def __hash__(self):
        return hash(self.route_index)

    def __eq__(self, other):
        return self.route_index == other.route_index

    def remove_customer(self, node: Node):
        assert node.is_depot is False, 'A depot is removed from a route'
        assert node in self._nodes, 'Node does not exist in route'
        self.size -= 1
        self.volume -= node.demand
        self._nodes.remove(node)
        if isinstance(self, RouteWithTW):
            _ = self.can_update_time_windows()

    def add_customers_after(self, nodes_to_add: list[Node], insert_after: Node):
        if insert_after not in self._nodes:
            raise ValueError(f"Customer {insert_after} not found in the route.")

        index = self._nodes.index(insert_after)
        self._nodes = self._nodes[:index + 1] + nodes_to_add + self._nodes[index + 1:]

        for node in nodes_to_add:
            assert node.is_depot is False, 'A depot is inserted into a route'
            self.size += 1
            self.volume += node.demand
        if isinstance(self, RouteWithTW):
            _ = self.can_update_time_windows()

    @property
    def customers(self) -> list[Node]:
        return self._nodes[1:-1]

    @property
    def nodes(self) -> list[Node]:
        return self._nodes[1:]

    @property
    def all_nodes(self) -> list[Node]:
        return self._nodes

    @property
    def edges(self) -> list[Edge]:
        return [
            Edge(self._nodes[idx], self._nodes[idx + 1])
            for idx in range(len(self._nodes) - 1)
        ]

    def validate(self):
        assert self._nodes[0].is_depot, 'First node has to be a depot.'
        assert self._nodes[-1].is_depot, 'Last node has to be a depot.'
        assert self._nodes[0] == self._nodes[-1], 'Start and return depot have to be the same'
        assert self.size == len(self._nodes) - 2
        assert self.volume == sum(node.demand for node in self._nodes)

        for node in self._nodes[1:-1]:
            assert node.is_depot == False

    def print(self) -> str:
        return '-'.join([str(node.node_id) for node in self._nodes])

    class SegmentAfter:
        def __init__(self, first_node, duration):
            self.first_node = first_node
            self.duration = duration

        @property
        def first(self):
            return self.first_node

    class SegmentBefore:
        def __init__(self, last_node, duration):
            self.last_node = last_node
            self.duration = duration

        @property
        def last(self):
            return self.last_node

    class SegmentMiddle:
        def __init__(self, first_node, last_node, duration):
            self.first_node = first_node
            self.last_node = last_node
            self.duration = duration

        @property
        def first(self):
            return self.first_node

        @property
        def last(self):
            return self.last_node


    class Proposal:
        def __init__(self, before_segment, middle_segments, after_segment, matrix):
            self.before = before_segment
            self.middle = middle_segments
            self.after = after_segment
            self.matrix = matrix

        def duration_segment(self):
            # Start with the segment before the change
            result = self.before._duration

            # For each node in the middle segment
            for node in self.middle:
                # Get travel time from last node to current node
                travel_time = self.matrix(result.last_node(), node.first())

                # Merge the segments
                result = DurationSegment.merge(travel_time, result, node._duration)

            # Connect with the segment after
            travel_time = self.matrix(result.last_node(), self.after.first_node())
            result = DurationSegment.merge(travel_time, result, self.after._duration)

            return result

        def distance_segment(self):
            # Similar to duration_segment but using DistanceSegment.merge
            # Implementation would follow the same pattern
            pass

        def load_segment(self):
            # Similar to duration_segment but using LoadSegment.merge
            # Implementation would follow the same pattern
            pass


class RouteWithTW(Route):
    """
    A route with time windows.
    """

    def __init__(self, nodes: List[NodeWithTW], route_index: int):
        super().__init__(nodes, route_index)

    def validate(self):
        super().validate()
        # make sure time window is valid (start < end)
        for i in range(1, len(self._nodes) - 1):
            node = self._nodes[i]
            assert node.time_window[0] <= node.time_window[1], \
                f"Time window error at node {node.node_id}: {node.time_window}"

    def can_update_time_windows(self) -> bool:
        """
        Updates the time windows of the nodes in the route based on the arrival times.
        """
        for i in range(1, len(self._nodes)):
            node = self._nodes[i]
            prev_node = self._nodes[i - 1]
            node_arrival_time = prev_node.arrival_time + prev_node.waiting_time + prev_node.service_time + compute_euclidean_distance(
                prev_node, node)
            if node_arrival_time > node.time_window[1]:
                return False
            if i == len(self._nodes) - 1:
                break
            node.arrival_time = node_arrival_time
            node.waiting_time = max(0, node.time_window[0] - node.arrival_time)

        return True


def compute_euclidean_distance(node1: Node, node2: Node) -> float:
    """
        Calculates Euclidean distance between two nodes.
        Returns int for CVRP problems (Node type) and float for VRPTW problems (NodeWithTW type).
        """
    distance = math.sqrt(
        math.pow(node1.x_coordinate - node2.x_coordinate, 2) +
        math.pow(node1.y_coordinate - node2.y_coordinate, 2)
    )

    # Return float for time windows instances (NodeWithTW), int otherwise
    if hasattr(node1, 'time_window') or hasattr(node2, 'time_window'):
        return distance
    else:
        return round(distance)
