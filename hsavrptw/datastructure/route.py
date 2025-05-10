import math
import sys
from typing import Optional

from datastructure.segment.DurationSegment import DurationSegment, Duration
from helpers.helpers import multiply_and_floor
from .edge import Edge
from .node import Node, NodeWithTW
from .segment.LoadSegment import LoadSegment, Load


class Route:
    MAX_CAPACITY: int

    def __init__(self, nodes: list[Node], route_index: int, max_capacity: int = 0):
        # Initialize the route with the depot as both the first and last node.
        assert nodes[0].is_depot, 'First node of a route has to be a depot.'
        assert nodes[-1].is_depot, 'Last node of a route has to be a depot.'
        assert nodes[0] == nodes[-1], 'Start and return depot has to be the same'

        self.route_index = route_index
        self.depot: Node = nodes[0]
        self._nodes: list = nodes.copy()

        self._curr: dict[int, Node] = {node.node_id: node for node in nodes}

        self.size = len(nodes) - 2  # Number of customers (not including depot)

        self.volume = sum(node.demand for node in self._nodes)  # Sum of demand of all customers of the route

        # Initialize duration segment lists
        self.dur_at: list[DurationSegment] = []
        self.dur_before: list[DurationSegment] = []
        self.dur_after: list[DurationSegment] = []

        # Initialize load segment lists
        self.load_at: list[LoadSegment] = []
        self.load_before: list[LoadSegment] = []
        self.load_after: list[LoadSegment] = []

        self._excess_load = Load(0)
        self.MAX_CAPACITY = max_capacity
        self.validate()

    def __repr__(self):
        return '-'.join([str(node.node_id) for node in self._nodes])

    def __hash__(self):
        return hash(self.route_index)

    def __eq__(self, other):
        return self.route_index == other._route_index

    def remove_customer(self, node: Node):
        assert node.is_depot is False, 'A depot is removed from a route'
        assert node in self._nodes, 'Node does not exist in route'
        self.size -= 1
        self.volume -= node.demand
        self._nodes.remove(node)
        if not hasattr(sys, "VRP_NO_TIME_WINDOWS"):
            # TODO: update time windows of route
            pass

    def add_customers_after(self, nodes_to_add: list[Node], insert_after: Node):
        if insert_after not in self._nodes:
            raise ValueError(f"Customer {insert_after} not found in the route.")

        index = self._nodes.index(insert_after)
        self._nodes = self._nodes[:index + 1] + nodes_to_add + self._nodes[index + 1:]

        for node in nodes_to_add:
            assert node.is_depot is False, 'A depot is inserted into a route'
            self.size += 1
            self.volume += node.demand

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

    def curr(self, ) -> dict[int, Node]:
        return self._curr

    @property
    def excess_load(self) -> Load:
        return Load(max(0, self.volume - self.MAX_CAPACITY))

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

    def update(self):
        """
        Update the route by recalculating the duration and load segments.
        """

        # Load
        self.load_at = [LoadSegment() for _ in range(len(self._nodes))]
        for i in range(1, len(self._nodes)):
            self.load_at[i] = LoadSegment(self._nodes[i].demand)

        self.load_before = [LoadSegment() for _ in range(len(self._nodes))]
        for i in range(1, len(self._nodes)):
            self.load_before[i] = LoadSegment.merge(self.load_before[i - 1], self.load_at[i])

        self.load_after = [LoadSegment() for _ in range(len(self._nodes))]
        for i in range(len(self._nodes) - 1, 0, -1):
            self.load_after[i - 1] = LoadSegment.merge(self.load_at[i - 1], self.load_after[i])

        # update volume and excess load
        self.volume = self.load_before[-1].load
        self._excess_load = self.excess_load

        # Duration segment (time windows)
        if not hasattr(sys, "VRP_NO_TIME_WINDOWS"):
            self.dur_at = [DurationSegment() for _ in range(len(self._nodes))]
            for i in range(len(self._nodes)):
                curr_node = self._nodes[i]
                self.dur_at[i] = DurationSegment(_duration=Duration(multiply_and_floor(curr_node.service_time)),
                                                 _time_warp=Duration(0),
                                                 _tw_early=Duration(multiply_and_floor(curr_node.time_window[0])),
                                                 _tw_late=Duration(multiply_and_floor(curr_node.time_window[1])),
                                                 _release_time=Duration(0))

            # precompute the forward duration segments
            self.dur_before = [DurationSegment() for _ in range(len(self._nodes))]
            self.dur_before[0] = self.dur_at[0]
            for i in range(1, len(self._nodes)):
                prev_node = self._nodes[i - 1]
                travel_time = compute_euclidean_distance(prev_node, self._nodes[i])
                self.dur_before[i] = DurationSegment.merge(Duration(travel_time), self.dur_before[i - 1],
                                                           self.dur_at[i])

            # precompute the backward duration segments
            self.dur_after = [DurationSegment() for _ in range(len(self._nodes))]
            self.dur_after[-1] = self.dur_at[-1]
            for i in range(len(self._nodes) - 1, 0, -1):
                prev_node = self._nodes[i - 1]
                travel_time = compute_euclidean_distance(prev_node, self._nodes[i])
                self.dur_after[i - 1] = DurationSegment.merge(Duration(travel_time), self.dur_at[i - 1],
                                                              self.dur_after[i])

    class SegmentBefore:
        """
        Class representing a route segment starting at the depot and ending at a specific node.

        Args:
            route: The route this segment belongs to
            end_index: The last node in the segment
        """

        def __init__(self, route, end_index):
            """
            Class representing a route segment starting at the depot and ending at a specific node.

            Args:
                route: The route this segment belongs to
                end_index: Index of the first node in the segment
            """
            self._route = route
            self._end_index = end_index

        @property
        def route(self):
            """Returns the route this segment belongs to"""
            return self._route

        @property
        def first(self) -> Node:
            """Returns the first node in the segment"""
            return self._route.all_nodes[0]

        @property
        def last(self) -> Node:
            """Returns the last node in the segment (depot)"""
            return self._route.all_nodes[self._end_index]


        @property
        def duration(self) -> Optional[DurationSegment]:
            """Calculate the duration of this segment with the given profile"""
            assert 0 <= self._end_index < len(self._route.dur_before), \
                f"Index {self._end_index} out of range for duration segments"
            return self._route.dur_before[self._end_index]

        @property
        def load(self) -> LoadSegment:
            """Return the load segment for this part of the route"""
            # Sum the demands of all nodes in the segment

            return self._route.load_before[self._end_index]

    class SegmentAfter:
        """
        Class representing a route segment starting at a specific node and ending at the depot.

        Args:
            route: The route this segment belongs to
            start_index: Index of the first node in the segment in the route
        """

        def __init__(self, route, start_index):
            """
            Initialize a segment from a starting node to the end of the route.

            Args:
                route: The route this segment belongs to
                start_index: Index of the first node in the segment
            """
            self._route = route
            self._start_index = start_index

        @property
        def route(self):
            """Returns the route this segment belongs to"""
            return self._route

        @property
        def first(self) -> Node:
            """Returns the first node in the segment"""
            return self._route.all_nodes[self._start_index]

        @property
        def last(self) -> Node:
            """Returns the last node in the segment (depot)"""
            return self._route.all_nodes[-1]

        @property
        def duration(self) -> Optional[DurationSegment]:
            """Calculate the duration of this segment with the given profile"""
            assert 0 <= self._start_index < len(self._route.dur_after), \
                f"Index {self._start_index} out of range for duration segments"

            return self._route.dur_after[self._start_index]

        @property
        def load(self) -> LoadSegment:
            """Return the load segment for this part of the route"""
            # Sum the demands of all nodes in the segment
            return self._route.load_after[self._start_index]

    class SegmentMiddle:
        """
        Class representing a route segment between two nodes.

        Args:
            route: The route this segment belongs to
            first_index: Index of the first node in the segment in the route
            last_index: Index of the last node in the segment in the route
        """
        _duration_segment: Optional[DurationSegment]
        _load_segment: LoadSegment

        def __init__(self, route, first_index, last_index):
            self._route = route
            self._first_index = first_index
            self._last_index = last_index

        @property
        def route(self):
            return self._route

        @property
        def first(self) -> Node:
            return self._route.all_nodes[self._first_index]

        @property
        def last(self) -> Node:
            return self._route.all_nodes[self._last_index]

        @property
        def duration(self) -> Optional[DurationSegment]:
            # Calculate the duration of this segment with the given profile
            assert 0 <= self._first_index <= self._last_index < len(self.route.dur_before), \
                f"Index start: {self._first_index}, end: {self._last_index} out of range for duration segments"

            self._duration_segment = self._route.dur_at[self._first_index]

            for i in range(self._first_index, self._last_index):
                # Merge the segments
                travel_time = compute_euclidean_distance(self.route.all_nodes[i], self.route.all_nodes[i + 1])
                self._duration_segment = DurationSegment.merge(Duration(travel_time), self._duration_segment,
                                                               self.route.dur_at[i + 1])

            return self._duration_segment

        @property
        def load(self) -> LoadSegment:
            # Return the load segment for this part of the route
            # Sum the demands of all nodes in the segment
            assert 0 <= self._first_index < self._last_index < len(self.route.load_before), \
                f"Index start: {self._first_index}, end: {self._last_index} out of range for load segments"

            self._load_segment = self._route.load_at[self._first_index]

            for i in range(self._first_index, self._last_index + 1):
                # Merge the segments
                self._load_segment = LoadSegment.merge(self._load_segment, self.route.load_at[i])

            return self._load_segment

    class Proposal:
        """
        Class representing a proposal for a route change.
        This class is used to calculate the duration and load segments for a proposed change in the route.
        The most important thing: it does not change the route itself, nor does it calculate the data on a new route.

        Args:
            before_segment: The segment before the change
            middle_segments: The segments in the middle of the change
            after_segment: The segment after the change

        """

        def __init__(self, before_segment, middle_segments, after_segment):
            self._before = before_segment
            self._middle = middle_segments
            self._after = after_segment

        @property
        def duration_segment(self) -> Optional[DurationSegment]:
            # Start with the segment before the change
            result = self._before.duration

            # Initialize travel time
            travel_time = compute_euclidean_distance(self._before.last, self._middle[0].first)

            # For each node in the middle segment
            for i in range(len(self._middle) - 1):
                # Each node is an actual segment that can be merged
                curr_segmt = self._middle[i]
                next_segmt = self._middle[i + 1]

                result = DurationSegment.merge(Duration(travel_time), result, curr_segmt.duration)
                travel_time = compute_euclidean_distance(curr_segmt.last, next_segmt.first)

            # Connect with the segment after
            travel_time = compute_euclidean_distance(self._middle[-1].last, self._after.first)
            result = DurationSegment.merge(Duration(travel_time), result, self._after.duration)

            return result

        @property
        def load_segment(self) -> LoadSegment:
            # Similar to duration_segment but using LoadSegment.merge

            result = self._before.load

            for i in range(len(self._middle) - 1):
                # Each node is an actual segment that can be merged
                result = LoadSegment.merge(result, self._middle[i].load)

            # Connect with the segment after
            result = LoadSegment.merge(result, self._after.load)
            return result

    def before(self, end_index) -> SegmentBefore:
        """
        Returns a segment of the route from the depot to the specified node.
        Args:
            end_index: The index of the last node in the segment
        """
        return self.SegmentBefore(self, end_index)

    def after(self, start_index) -> SegmentAfter:
        """
        Returns a segment of the route from the specified node to the depot.
        Args:
            start_index: The index of the first node in the segment
        """
        return self.SegmentAfter(self, start_index)

    def between(self, start_index, end_index) -> SegmentMiddle:
        """
        Returns a segment of the route between two specified nodes.
        Args:
            start_index: The index of the first node in the segment
            end_index: The index of the last node in the segment
        """
        return self.SegmentMiddle(self, start_index, end_index)

    def at(self, index) -> SegmentMiddle:
        """
        Returns a segment of the route at the specified index.
        Args:
            index: The index of the node in the segment
        """
        return self.SegmentMiddle(self, index, index)

    def proposal(self, before_segment: SegmentBefore, middle_segments: list[SegmentMiddle],
                 after_segment: SegmentAfter) -> Proposal:
        """
        Returns a proposal for a route change.
        """
        return self.Proposal(before_segment, middle_segments, after_segment)


# class RouteWithTW(Route):
#     """
#     A route with time windows.
#     """
#
#     def __init__(self, nodes: list[NodeWithTW], route_index: int):
#         super().__init__(nodes, route_index)
#
#     def validate(self):
#         super().validate()
#         # make sure time window is valid (start < end)
#         for i in range(1, len(self._nodes) - 1):
#             node = self._nodes[i]
#             assert node.time_window[0] <= node.time_window[1], \
#                 f"Time window error at node {node.node_id}: {node.time_window}"
#
#     def can_update_time_windows(self) -> bool:
#         """
#         Updates the time windows of the nodes in the route based on the arrival times.
#         """
#         for i in range(1, len(self._nodes)):
#             node = self._nodes[i]
#             prev_node = self._nodes[i - 1]
#             node_arrival_time = prev_node.arrival_time + prev_node.waiting_time + prev_node.service_time + compute_euclidean_distance(
#                 prev_node, node)
#             if node_arrival_time > node.time_window[1]:
#                 return False
#             if i == len(self._nodes) - 1:
#                 break
#             node.arrival_time = node_arrival_time
#             node.waiting_time = max(0, node.time_window[0] - node.arrival_time)
#
#         return True


def compute_euclidean_distance(node1: Node, node2: Node) -> int:
    """
        Calculates Euclidean distance between two nodes.
        Returns int for CVRP problems (Node type) and float for VRPTW problems (NodeWithTW type).
    """
    distance = math.sqrt(
        math.pow(node1.x_coordinate - node2.x_coordinate, 2) +
        math.pow(node1.y_coordinate - node2.y_coordinate, 2)
    )
    return multiply_and_floor(distance, 1)
