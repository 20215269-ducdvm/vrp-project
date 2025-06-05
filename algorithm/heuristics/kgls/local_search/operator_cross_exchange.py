import logging

from datastructure import Node, Route, VRPSolution, CostEvaluator
# from datastructure.cost_evaluator import Cost
from .local_search_move import LocalSearchMove

logger = logging.getLogger(__name__)


class CrossExchange(LocalSearchMove):

    def __init__(
            self,
            segment1: list[Node],
            segment2: list[Node],
            segment1_insert_after: Node,
            segment2_insert_after: Node,
            route1: Route,
            route2: Route,
            improvement: int,
            start_node: Node
    ):
        self.segment1 = segment1
        self.segment2 = segment2
        self.segment1_insert_after = segment1_insert_after
        self.segment2_insert_after = segment2_insert_after
        self.start_node = start_node
        self.route1 = route1
        self.route2 = route2

        self.improvement = improvement

    def get_routes(self) -> list[Route]:
        return set([self.route1, self.route2])

    def is_disjunct(self, other):
        if self.route1 in other.get_routes():
            return False
        if self.route2 in other.get_routes():
            return False

        return True

    def execute(self, solution: VRPSolution):
        logger.debug(f'Executing cross-exchange with segments of sizes '
                     f'{len(self.segment1)} and {len(self.segment2)} '
                     f'with improvement of {int(self.improvement)}')

        solution.remove_nodes(self.segment1)
        solution.remove_nodes(self.segment2)

        solution.insert_nodes_after(self.segment1, self.segment1_insert_after, self.route2)
        solution.insert_nodes_after(self.segment2, self.segment2_insert_after, self.route1)

    def print_log(self):
        print(
            f'CrossExchange: {self.start_node.node_id} '
            f'segment1: {self.segment1} '
            f'segment2: {self.segment2} '
            f'improvement: {self.improvement}'
        )


def search_cross_exchanges_from(
        solution: VRPSolution,
        cost_evaluator: CostEvaluator,
        start_node: Node,
        segment1_directions: list[int, int] = [0, 1],
        segment2_directions: list[int, int] = [0, 1],
) -> list[CrossExchange]:
    # try to exchange a node segment starting with start_node (and extending it into 'direction')
    # with a segment from another route, starting from a neighborhood node of 'start_node'
    route1: Route = solution.route_of(start_node)

    # get the penalties of route1
    dist1, load1, tw1 = solution.get_route_penalties(route1)
    route1_cost = cost_evaluator.get_route_cost(dist1, load1, tw1)
    candidate_moves: list[CrossExchange] = []

    for segment1_direction in segment1_directions:
        for segment2_direction in segment2_directions:
            # find the node after start_node
            # this is the start of the segment in route1 that we want to exchange with route2
            route1_segment_connection_start = solution.neighbour(start_node, 1 - segment1_direction)

            # start searching for nearby nodes of start_node that don't belong to the same route
            # from these nodes, we will try to find segments that can be exchanged with the segment in route1
            for route2_segment_connection_start in cost_evaluator.get_neighborhood(start_node):
                # find the route of each neighboring node
                route2 = solution.route_of(route2_segment_connection_start)
                dist2, load2, tw2 = solution.get_route_penalties(route2)
                route2_cost = cost_evaluator.get_route_cost(dist2, load2, tw2)

                # if the neighboring node is in a different route than start_node
                if route2 != route1:
                    # find the node after route2_segment_connection_start
                    # this is the start of the segment in route2 that we want to exchange with route1
                    segment2_start = solution.neighbour(route2_segment_connection_start, segment2_direction)
                    if segment2_start.is_depot:
                        continue

                    # at this point, we have:
                    # - start_node: the start node of the segment in route1 that we want to exchange
                    # - route1_segment_connection_start: the node after start_node in route1
                    # - route2_segment_connection_start: neighbor node of start_node in route2
                    # - segment2_start: the node we want to exchange with start_node in route2
                    # - route1: ... - start_node - route1_segment_connection_start - ...
                    # - route2: ... - route2_segment_connection_start - segment2_start - ...

                    # check if the first cross-exchange is improving:
                    # - route1 proposal: ... - start_node - route2_segment_connection_start - ...
                    # - route2 proposal: ... - route1_segment_connection_start - segment2_start - ...

                    improvement_first_cross = (
                            cost_evaluator.get_distance(start_node, route1_segment_connection_start)
                            + cost_evaluator.get_distance(route2_segment_connection_start, segment2_start)
                            - cost_evaluator.get_distance(start_node, route2_segment_connection_start)
                            - cost_evaluator.get_distance(route1_segment_connection_start, segment2_start)
                    )

                    # if the first cross-exchange is improving, we can start extending from segment2_start
                    # the segments until we reach a depot or the capacity of the routes is violated

                    if improvement_first_cross > 0:
                        # Initialize segments - segment1 starts with start_node
                        segment1_end = start_node
                        # complete list of segment1
                        segment1_list = [segment1_end]
                        # indices of start and end of segment1 in route1
                        segment1_start_idx = route1.idx_of[start_node.node_id]
                        segment1_end_idx = segment1_start_idx

                        # try to extend segment 1 until the end
                        while not segment1_end.is_depot:
                            # extend segment2 until capacity and time of route 1 is violated
                            segment2_end = segment2_start
                            # complete list of segment2
                            segment2_list = [segment2_end]
                            # indices of start and end of segment2 in route2
                            segment2_start_idx = route2.idx_of[segment2_start.node_id]
                            segment2_end_idx = segment2_start_idx

                            # now we'll change
                            # route1: ... - start_node - segment1_end - ...
                            # route2: ... - route2_segment_connection_start - segment2_start - segment2_end - ...
                            # to
                            # route1: ... - start_node - segment2_start - segment2_end - ...
                            # route2: ... - route2_segment_connection_start - segment1_end - ...
                            # create a cross-exchange proposal for route 1
                            proposal_route1 = route1.proposal(
                                before_segment=route1.before(segment1_start_idx - 1),
                                middle_segments=[
                                    route1.at(segment1_start_idx),
                                    route2.at(segment2_start_idx)
                                ],
                                after_segment=route1.after(segment1_start_idx + 1)
                            )

                            # get the cost of the proposal for route 1
                            proposal_route1_cost = cost_evaluator.get_route_cost(proposal_route1.penalties[0],
                                                                                 proposal_route1.penalties[1],
                                                                                 proposal_route1.penalties[2])

                            # check: # if the proposal for route 1 is feasible or
                            # if it is not feasible but the cost is lower than the current cost
                            while (not segment2_end.is_depot and
                                   # proposal_route1_cost > route1_cost
                                   (proposal_route1.is_feasible or (
                                           not proposal_route1.is_feasible and
                                           proposal_route1_cost > route1_cost
                                   ))
                            ):
                                proposal_route2 = route2.proposal(
                                    before_segment=route2.before(segment2_start_idx - 1),
                                    middle_segments=[
                                        route2.at(segment2_start_idx),
                                        route1.at(segment1_end_idx)
                                    ],
                                    after_segment=route2.after(segment2_end_idx + 1)
                                )
                                proposal_route2_cost = cost_evaluator.get_route_cost(
                                    proposal_route2.penalties[0],
                                    proposal_route2.penalties[1],
                                    proposal_route2.penalties[2]
                                )
                                # check feasibility of route 2
                                if (
                                        proposal_route2.is_feasible or (
                                        not proposal_route2.is_feasible and
                                        proposal_route2_cost > route2_cost
                                )
                                ):
                                    # check overall improvement of move
                                    # route1_segment_connection_end = segment1_end.get_neighbour(segment1_direction)
                                    # route2_segment_connection_end = segment2_end.get_neighbour(segment2_direction)
                                    route1_segment_connection_end = solution.neighbour(segment1_end, segment1_direction)
                                    route2_segment_connection_end = solution.neighbour(segment2_end, segment2_direction)

                                    improvement_second_cross = (
                                            cost_evaluator.get_distance(segment1_end, route1_segment_connection_end)
                                            + cost_evaluator.get_distance(segment2_end, route2_segment_connection_end)
                                            - cost_evaluator.get_distance(segment1_end, route2_segment_connection_end)
                                            - cost_evaluator.get_distance(segment2_end, route1_segment_connection_end)
                                    )
                                    improvement = improvement_first_cross + improvement_second_cross

                                    if improvement > 0:
                                        # store move
                                        candidate_moves.append(
                                            CrossExchange(
                                                segment1=segment1_list.copy(),
                                                segment2=segment2_list.copy(),
                                                route1=route1,
                                                route2=route2,
                                                segment1_insert_after=route2_segment_connection_start if segment2_direction == 1 else route2_segment_connection_end,
                                                segment2_insert_after=route1_segment_connection_start if segment1_direction == 1 else route1_segment_connection_end,
                                                improvement=improvement,
                                                start_node=start_node
                                            )
                                        )

                                # extend segment2
                                # segment lists are in the order as the nodes are later inserted
                                # segment2_end = segment2_end.get_neighbour(segment2_direction)
                                segment2_end = solution.neighbour(segment2_end, segment2_direction)

                                if (segment2_direction == 1 and segment1_direction == 0) or (
                                        segment1_direction + segment2_direction == 0):
                                    segment2_list.insert(0, segment2_end)
                                else:
                                    segment2_list.append(segment2_end)

                            # extend segment1
                            # segment1_end = segment1_end.get_neighbour(segment1_direction)
                            segment1_end = solution.neighbour(segment1_end, segment1_direction)
                            if (segment1_direction == 1 and segment2_direction == 0) or (
                                    segment1_direction + segment2_direction == 0):
                                segment1_list.insert(0, segment1_end)
                            else:
                                segment1_list.append(segment1_end)

    return candidate_moves


def search_cross_exchanges(
        solution: VRPSolution,
        cost_evaluator: CostEvaluator,
        start_nodes: list[Node],
) -> list[CrossExchange]:
    candidate_moves = []
    for start_node in start_nodes:
        candidate_moves.extend(
            search_cross_exchanges_from(solution, cost_evaluator, start_node)
        )

    return sorted(candidate_moves)
