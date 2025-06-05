import logging

from datastructure import Node, Route, VRPSolution, CostEvaluator
from .local_search_move import LocalSearchMove

logger = logging.getLogger(__name__)


class SegmentMove(LocalSearchMove):

    def __init__(
            self,
            segment: list[Node],
            from_route: Route,
            to_route: Route,
            move_after: Node,
            improvement: float
    ):
        self.segment: list[Node] = segment
        self.from_route = from_route
        self.to_route = to_route
        self.move_after: Node = move_after
        self.improvement: float = improvement

    def get_routes(self) -> set[Route]:
        return set([self.from_route, self.to_route])

    def is_disjunct(self, other):
        if self.from_route in other.get_routes():
            return False
        if self.to_route in other.get_routes():
            return False

        return True

    def execute(self, solution: VRPSolution):
        logger.debug(f'Executing Segment relocation with segment of size {len(self.segment)} '
                     f'with improvement of {int(self.improvement)}')

        solution.remove_nodes(self.segment)
        solution.insert_nodes_after(self.segment, self.move_after, self.to_route)


def search_3_opt_moves_from(
        solution: VRPSolution,
        cost_evaluator: CostEvaluator,
        start_node: Node,
        segment_directions: list[int] = [0, 1],
        insert_directions: list[int] = [0, 1]
) -> list[SegmentMove]:
    candidate_moves: list[SegmentMove] = []
    from_route = solution.route_of(start_node)

    # Get the penalties of from_route
    dist1, load1, tw1 = solution.get_route_penalties(from_route)
    from_route_cost = cost_evaluator.get_route_cost(dist1, load1, tw1)

    for segment_direction in segment_directions:
        for insert_direction in insert_directions:

            segment_1_prev = solution.neighbour(start_node, 1 - segment_direction)

            for insert_next_to in cost_evaluator.get_neighborhood(start_node):
                to_route = solution.route_of(insert_next_to)

                # Get the penalties of to_route
                dist2, load2, tw2 = solution.get_route_penalties(to_route)
                to_route_cost = cost_evaluator.get_route_cost(dist2, load2, tw2)

                if to_route != from_route:
                    # compute improvement of first edge change
                    insert_next_to_2 = solution.neighbour(insert_next_to, insert_direction)

                    move_start_improvement = (
                            cost_evaluator.get_distance(start_node, segment_1_prev)
                            + cost_evaluator.get_distance(insert_next_to, insert_next_to_2)
                            - cost_evaluator.get_distance(insert_next_to, start_node)
                    )

                    if move_start_improvement > 0:
                        segment_end = start_node
                        segment_list = [segment_end]
                        segment_start_idx = from_route.idx_of[start_node.node_id]
                        segment_end_idx = segment_start_idx

                        while not segment_end.is_depot:
                            segment_disconnect_2 = solution.neighbour(segment_end, segment_direction)

                            move_end_improvement = (
                                    cost_evaluator.get_distance(segment_end, segment_disconnect_2)
                                    - cost_evaluator.get_distance(segment_1_prev, segment_disconnect_2)
                                    - cost_evaluator.get_distance(segment_end, insert_next_to_2)
                            )

                            improvement = move_start_improvement + move_end_improvement

                            # Create proposal for from_route with the segment removed
                            segment_prev_idx = segment_start_idx - 1 if segment_start_idx > 0 else 0
                            segment_next_idx = segment_end_idx + 1 if segment_end_idx < len(from_route.nodes) - 1 else len(from_route.nodes) - 1

                            proposal_from_route = from_route.proposal(
                                before_segment=from_route.before(segment_prev_idx),
                                middle_segments=[],  # Skip the segment
                                after_segment=from_route.after(segment_next_idx)
                            )

                            # Create proposal for to_route with the segment inserted
                            insert_after_idx = to_route.idx_of[insert_next_to.node_id]
                            print(insert_after_idx)
                            proposal_to_route = to_route.proposal(
                                before_segment=to_route.before(insert_after_idx),
                                middle_segments=[
                                    # Add the segment as a middle segment
                                    from_route.between(segment_start_idx, segment_end_idx)
                                ],
                                after_segment=to_route.after(insert_after_idx + 1 if insert_after_idx < len(to_route.nodes) - 1 else insert_after_idx)
                            )

                            proposal_from_route_cost = cost_evaluator.get_route_cost(
                                proposal_from_route.penalties[0],
                                proposal_from_route.penalties[1],
                                proposal_from_route.penalties[2]
                            )

                            proposal_to_route_cost = cost_evaluator.get_route_cost(
                                proposal_to_route.penalties[0],
                                proposal_to_route.penalties[1],
                                proposal_to_route.penalties[2]
                            )

                            if improvement > 0 and (
                                    (proposal_from_route.is_feasible and proposal_to_route.is_feasible) or
                                    (
                                            proposal_from_route_cost > from_route_cost and proposal_to_route_cost > to_route_cost)
                            ):
                                # store move
                                if insert_direction == 1:
                                    insert_after = insert_next_to
                                else:
                                    insert_after = insert_next_to_2

                                candidate_moves.append(
                                    SegmentMove(
                                        segment=segment_list.copy(),
                                        from_route=from_route,
                                        to_route=to_route,
                                        move_after=insert_after,
                                        improvement=improvement,
                                        start_node=start_node
                                    )
                                )

                            # extend
                            segment_end = segment_disconnect_2
                            segment_end_idx += 1
                            if segment_direction == 0:
                                segment_list.append(segment_end)
                            else:
                                segment_list.insert(0, segment_end)

                            if segment_end.is_depot:
                                break

    return candidate_moves


def search_3_opt_moves(
        solution: VRPSolution,
        cost_evaluator: CostEvaluator,
        start_nodes: list[Node],
) -> list[SegmentMove]:
    candidate_moves = []
    for start_node in start_nodes:
        candidate_moves.extend(
            search_3_opt_moves_from(solution, cost_evaluator, start_node)
        )

    return sorted(candidate_moves)


