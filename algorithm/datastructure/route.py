import sys
from typing import Optional

from datastructure.segment.DurationSegment import DurationSegment, Duration
from helpers.helpers import multiply_and_floor, compute_euclidean_distance
from .edge import Edge
from .node import Node
from .segment.DistanceSegment import DistanceSegment, Distance
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

        # self._curr: dict[int, Node] = {node.node_id: node for node in nodes}
        self.idx_of: dict[int, int] = {node.node_id: idx for idx, node in
                                       enumerate(nodes[:-1])}  # Exclude the last depot node

        self.size = len(nodes) - 2  # Number of customers (not including depot)

        self.volume = sum(node.demand for node in self._nodes)  # Sum of demand of all customers of the route

        # Initialize distance segment lists
        self.dist_at: list[DistanceSegment] = []
        self.dist_before: list[DistanceSegment] = []
        self.dist_after: list[DistanceSegment] = []

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
        self._time_warp = None

        self.validate()
        self.update()

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

    @property
    def total_distance(self) -> int:
        """
        Calculate the total distance of the route.
        """
        return self.dist_before[-1].distance if self.dist_before else 0

    @property
    def excess_load(self) -> int:
        return Load(max(0, self.volume - self.MAX_CAPACITY))

    @property
    def time_warp(self) -> int:
        return self._time_warp

    @property
    def is_feasible(self) -> bool:
        """
        Check if the route is feasible based on the current load and capacity.
        """
        return self._excess_load <= 0 and self._time_warp == 0

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

        self.idx_of: dict[int, int] = {node.node_id: idx for idx, node in
                                       enumerate(self._nodes[:-1])}  # Exclude the last depot node

        # Distance
        self.dist_at = [DistanceSegment() for _ in range(len(self._nodes))]
        for i in range(1, len(self._nodes)):
            self.dist_at[i] = DistanceSegment(Distance(0))

        # precompute the forward distance segments
        self.dist_before = [DistanceSegment() for _ in range(len(self._nodes))]
        self.dist_before[0] = self.dist_at[0]
        for i in range(1, len(self._nodes)):
            travel_dist = compute_euclidean_distance(self._nodes[i - 1], self._nodes[i])
            self.dist_before[i] = DistanceSegment.merge(Distance(travel_dist), self.dist_before[i - 1], self.dist_at[i])

        # precompute the backward distance segments
        self.dist_after = [DistanceSegment() for _ in range(len(self._nodes))]
        self.dist_after[-1] = self.dist_at[-1]
        for i in range(len(self._nodes) - 1, 0, -1):
            travel_dist = compute_euclidean_distance(self._nodes[i - 1], self._nodes[i])
            self.dist_after[i - 1] = DistanceSegment.merge(Distance(travel_dist), self.dist_at[i - 1],
                                                           self.dist_after[i])

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

            self._time_warp = self.dur_before[-1].time_warp()

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
        def distance(self) -> DistanceSegment:
            """Calculate the distance of this segment with the given profile"""
            assert 0 <= self._end_index < len(self._route.dist_before), \
                f"Index {self._end_index} out of range for distance segments"
            return self._route.dist_before[self._end_index]

        @property
        def duration(self) -> DurationSegment:
            """Calculate the duration of this segment with the given profile"""
            if hasattr(sys, "VRP_NO_TIME_WINDOWS"):
                return DurationSegment()
            assert 0 <= self._end_index < len(self._route.dur_before), \
                f"Index {self._end_index} out of range for duration segments"
            return self._route.dur_before[self._end_index]

        @property
        def load(self) -> LoadSegment:
            """Return the load segment for this part of the route"""
            # Sum the demands of all nodes in the segment
            assert 0 <= self._end_index < len(self._route.load_before), \
                f"Index {self._end_index} out of range for load segments"
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
            # print(self._start_index)
            return self._route.all_nodes[self._start_index]

        @property
        def last(self) -> Node:
            """Returns the last node in the segment (depot)"""
            return self._route.all_nodes[-1]

        @property
        def distance(self) -> DistanceSegment:
            """Calculate the distance of this segment with the given profile"""
            assert 0 <= self._start_index < len(self._route.dist_after), \
                f"Index {self._start_index} out of range for distance segments"

            return self._route.dist_after[self._start_index]

        @property
        def duration(self) -> DurationSegment:
            """Calculate the duration of this segment with the given profile"""
            if hasattr(sys, "VRP_NO_TIME_WINDOWS"):
                return DurationSegment()

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
        def distance(self) -> DistanceSegment:
            # Calculate the distance of this segment with the given profile
            print(self._first_index, self._last_index)
            assert 0 <= self._first_index <= self._last_index < len(self.route.dist_before), \
                f"Index start: {self._first_index}, end: {self._last_index} out of range for distance segments"

            self._distance_segment = self._route.dist_at[self._first_index]

            for i in range(self._first_index, self._last_index):
                # Merge the segments
                travel_dist = compute_euclidean_distance(self.route.all_nodes[i], self.route.all_nodes[i + 1])
                self._distance_segment = DistanceSegment.merge(Distance(travel_dist), self._distance_segment,
                                                               self.route.dist_at[i + 1])

            return self._distance_segment

        @property
        def duration(self) -> DurationSegment:
            # Calculate the duration of this segment with the given profile
            if hasattr(sys, "VRP_NO_TIME_WINDOWS"):
                return DurationSegment()

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
            assert 0 <= self._first_index <= self._last_index < len(self.route.load_before), \
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
            before_segment: The segment before the changed segment
            middle_segments: The segments in the middle of the change
            after_segment: The segment after the changed segment

        """

        def __init__(self, route, before_segment, middle_segments, after_segment):
            self._route = route
            self._before = before_segment
            self._middle = middle_segments
            self._after = after_segment

        @property
        def distance_segment(self) -> DistanceSegment:
            if len(self._middle) == 0:
                # If there are no middle segments, just return the distance segment from before and after
                return DistanceSegment.merge(
                    Distance(compute_euclidean_distance(self._before.last, self._after.first)),
                    self._before.distance,
                    self._after.distance)

            result = self._before.distance

            # Initialize travel distance
            travel_distance = compute_euclidean_distance(self._before.last, self._middle[0].first)

            # For each node in the middle segment
            for i in range(len(self._middle)):
                # Each node is an actual segment that can be merged
                curr_segmt = self._middle[i]
                result = DistanceSegment.merge(Distance(travel_distance), result, curr_segmt.distance)

                if i == len(self._middle) - 1:
                    # Last segment, connect with the segment after
                    travel_distance = compute_euclidean_distance(curr_segmt.last, self._after.first)
                    break

                next_segmt = self._middle[i + 1]
                travel_distance = compute_euclidean_distance(curr_segmt.last, next_segmt.first)

            # Connect with the segment after
            result = DistanceSegment.merge(Distance(travel_distance), result, self._after.distance)

            return result

        @property
        def duration_segment(self) -> DurationSegment:
            if hasattr(sys, "VRP_NO_TIME_WINDOWS"):
                # If time windows are disabled, return None
                return DurationSegment()

            if len(self._middle) == 0:
                # If there are no middle segments, just return the duration segment from before and after
                return DurationSegment.merge(
                    Duration(compute_euclidean_distance(self._before.last, self._after.first)),
                    self._before.duration,
                    self._after.duration)

            # Start with the segment before the change
            result = self._before.duration

            # Initialize travel time
            travel_time = compute_euclidean_distance(self._before.last, self._middle[0].first)

            # For each node in the middle segment
            for i in range(len(self._middle)):
                # Each node is an actual segment that can be merged
                curr_segmt = self._middle[i]
                result = DurationSegment.merge(Duration(travel_time), result, curr_segmt.duration)

                if i == len(self._middle) - 1:
                    # Last segment, connect with the segment after
                    travel_time = compute_euclidean_distance(curr_segmt.last, self._after.first)
                    break

                next_segmt = self._middle[i + 1]
                travel_time = compute_euclidean_distance(curr_segmt.last, next_segmt.first)

            # Connect with the segment after
            # travel_time = compute_euclidean_distance(self._middle[-1].last, self._after.first)
            result = DurationSegment.merge(Duration(travel_time), result, self._after.duration)

            return result

        @property
        def load_segment(self) -> LoadSegment:
            if len(self._middle) == 0:
                # If there are no middle segments, just return the load segment from before and after
                return LoadSegment.merge(self._before.load, self._after.load)

            # Similar to duration_segment but using LoadSegment.merge
            result = self._before.load

            for i in range(len(self._middle)):
                # Each node is an actual segment that can be merged
                result = LoadSegment.merge(result, self._middle[i].load)

            # Connect with the segment after
            result = LoadSegment.merge(result, self._after.load)
            return result

        @property
        def penalties(self) -> tuple[Distance, Load, Duration]:
            """
            Returns the penalties of the proposal.
            """
            return (self.distance_segment.distance, self.load_segment.excess_load(self._route.MAX_CAPACITY),
                    self.duration_segment.time_warp())

        @property
        def is_feasible(self) -> bool:
            """
            Check if the proposal is feasible based on the load and time warp.
            """
            return (self.load_segment.load <= self._route.MAX_CAPACITY and
                    self.duration_segment.time_warp() == Duration(0))

    def before(self, end_index) -> SegmentBefore:
        """
        Returns a segment of the route from the depot (beginning) to the specified node.
        Args:
            end_index: The index of the last node in the segment
        """
        return self.SegmentBefore(self, end_index)

    def after(self, start_index) -> SegmentAfter:
        """
        Returns a segment of the route from the specified node to the depot (end).
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
        return self.Proposal(self, before_segment, middle_segments, after_segment)
