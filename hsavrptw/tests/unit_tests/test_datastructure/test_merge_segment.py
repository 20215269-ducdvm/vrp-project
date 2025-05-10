import unittest

from datastructure import Route
from datastructure.segment.DurationSegment import Duration
from tests.unit_tests.test_local_search_operators.test_cross_exchange import build_toy_problem_vrptw


class TestMergeSegment(unittest.TestCase):
    def test_merge_intra_route(self):
        problem, evaluator = build_toy_problem_vrptw()
        nodes = [
            problem.nodes[0],
            problem.nodes[4],
            problem.nodes[5],
            problem.nodes[6],
            problem.nodes[0],
        ]
        # route: [0, 4, 5, 6, 0]
        route: Route = Route(nodes=nodes, route_index=0, max_capacity=problem.capacity)
        route.update()

        # Test 1: New route: [0, 4, 5, 6, 0] -> Proposed route: [0, 6, 5, 4, 0]
        proposal: Route.Proposal = route.proposal(
            before_segment=route.before(0), # 0
            middle_segments=[
                route.at(3), # 6
                route.at(2), # 5
                route.at(1), # 4
            ],
            after_segment=route.after(4), # 0
        )
        # End result: Total duration of new route: 0, 6, 5, 4, 0, without actually creating and compute on it.
        duration_segment = proposal.duration_segment
        assert duration_segment.time_warp() == Duration(0)

    def test_merge_inter_route(self):
        problem, evaluator = build_toy_problem_vrptw()
        nodes1 = [
            problem.nodes[0],
            problem.nodes[3],
            problem.nodes[5],
            problem.nodes[4],
            problem.nodes[0],
        ]

        # route1: [0, 3, 5, 4, 0]
        route1: Route = Route(nodes=nodes1, route_index=0, max_capacity=problem.capacity)

        nodes2 = [
            problem.nodes[0],
            problem.nodes[6],
            problem.nodes[1],
            problem.nodes[2],
            problem.nodes[0],
        ]

        # route2: [0, 6, 1, 2, 0]
        route2: Route = Route(nodes=nodes2, route_index=1, max_capacity=problem.capacity)


        # Test 2: Propose to swap segments [5, 4] in route1 and [1, 2] in route2
        # route1: [0, 3, 5, 4, 0] -> Proposed route: [0, 3, 1, 2, 0]
        # route2: [0, 6, 1, 2, 0] -> Proposed route: [0, 6, 5, 4, 0]

        route1.update()
        route2.update()
        proposal_for_route1: Route.Proposal = route1.proposal(
            before_segment=route1.before(1), # Duration of segment 0, 3
            middle_segments=[
                route2.between(2, 3), # Combine with duration of segment [1, 2]
            ],
            after_segment=route2.after(4), # Combine with duration of segment 0
        )
        proposal_for_route2: Route.Proposal = route2.proposal(
            before_segment=route2.before(1), # Duration of segment 0, 6
            middle_segments=[
                route1.between(2, 3), # Combine with duration of segment [5, 4]
            ],
            after_segment=route2.after(4),  # Combine with duration of segment 0
        )

        duration_segment_route1 = proposal_for_route1.duration_segment
        assert duration_segment_route1.time_warp() == Duration(0)

        duration_segment_route2 = proposal_for_route2.duration_segment
        assert duration_segment_route2.time_warp() == Duration(0)
