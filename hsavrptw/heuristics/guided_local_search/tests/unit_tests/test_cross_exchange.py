
from kgls.local_search.operator_cross_exchange import search_cross_exchanges_from
from kgls.datastructure import Node, VRPProblem, CostEvaluator, VRPSolution


def build_problem() -> tuple[VRPProblem, CostEvaluator]:
    # Depot in the middle
    # 1   2   3
    #     D
    #     4   5
    depot = Node(node_id=0, x_coordinate=50, y_coordinate=20, demand=0, is_depot=True)
    customers = [
        Node(node_id=1, x_coordinate=0, y_coordinate=10, demand=1, is_depot=False),
        Node(node_id=2, x_coordinate=0, y_coordinate=20, demand=1, is_depot=False),
        Node(node_id=3, x_coordinate=0, y_coordinate=30, demand=1, is_depot=False),
        Node(node_id=4, x_coordinate=100, y_coordinate=10, demand=1, is_depot=False),
        Node(node_id=5, x_coordinate=100, y_coordinate=20, demand=1, is_depot=False),
    ]
    all_nodes = [depot] + customers

    vrp_problem = VRPProblem(all_nodes, 3)
    vrp_evaluator = CostEvaluator(all_nodes, 3, {'neighborhood_size': 5})

    return vrp_problem, vrp_evaluator


def test_search_cross_exchanges_from():

    problem, evaluator = build_problem()
    # optimal routes are D-1-2-3-D and D-4-5-D

    # CASE 1: exchange node 4 and node 2
    route1 = [
        problem.nodes[1],
        problem.nodes[4],
        problem.nodes[3],
    ]
    route2 = [
        problem.nodes[2],
        problem.nodes[5],
    ]
    solution = VRPSolution(problem)
    solution.add_route(route1)
    solution.add_route(route2)

    found_moves: list = search_cross_exchanges_from(
        solution=solution,
        cost_evaluator=evaluator,
        start_node=problem.nodes[4]
    )

    assert found_moves
    best_move = sorted(found_moves)[0]
    assert best_move.improvement == 271
    assert best_move.segment1 == [problem.nodes[4]]
    assert best_move.segment2 == [problem.nodes[2]]

    # CASE 3: exchange [2,3] and [4]
    route1 = [
        problem.nodes[1],
        problem.nodes[4],
    ]
    route2 = [
        problem.nodes[5],
        problem.nodes[2],
        problem.nodes[3],
    ]
    solution = VRPSolution(problem)
    solution.add_route(route1)
    solution.add_route(route2)

    found_moves: list = search_cross_exchanges_from(
        solution=solution,
        cost_evaluator=evaluator,
        start_node=problem.nodes[2]
    )

    assert found_moves
    best_move = sorted(found_moves)[0]
    assert best_move.improvement == 180
    assert best_move.segment1 == [problem.nodes[2], problem.nodes[3]]
    assert best_move.segment2 == [problem.nodes[4]]

    # CASE 3: exchange [3,2] and [4], starting from node3
    # Note that the segment [3,2] is traversed reversely in the current route
    route1 = [
        problem.nodes[1],
        problem.nodes[4],
    ]
    route2 = [
        problem.nodes[5],
        problem.nodes[2],
        problem.nodes[3],
    ]
    solution = VRPSolution(problem)
    solution.add_route(route1)
    solution.add_route(route2)

    found_moves: list = search_cross_exchanges_from(
        solution=solution,
        cost_evaluator=evaluator,
        start_node=problem.nodes[3]
    )

    assert found_moves
    best_move = sorted(found_moves)[0]
    assert best_move.improvement == 171
    assert best_move.segment1 == [problem.nodes[3], problem.nodes[2]]
    assert best_move.segment2 == [problem.nodes[4]]
