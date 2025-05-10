from datastructure import VRPSolution, VRPProblem, CostEvaluator, Node
from datastructure.node import NodeWithTW
from heuristics.kgls.local_search.operator_cross_exchange import search_cross_exchanges_from


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

    # CASE 2: exchange [2,3] and [4]
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

def test_search_cross_exchanges_from_tw():
    problem, evaluator = build_toy_problem_vrptw()
    # optimal routes are D-3-1-2-D and D-6-5-4-D

    # CASE 1: exchange node 5 and node 1
    route1 = [
        problem.nodes[3],
        problem.nodes[5],
        problem.nodes[2],
    ]
    route2 = [
        problem.nodes[6],
        problem.nodes[1],
        problem.nodes[4],
    ]
    solution = VRPSolution(problem)
    solution.add_route(route1)
    solution.add_route(route2)

    found_moves: list = search_cross_exchanges_from(
        solution=solution,
        cost_evaluator=evaluator,
        start_node=problem.nodes[5]
    )

    assert found_moves
    best_move = sorted(found_moves)[0]
    # assert best_move.improvement == 0
    assert best_move.segment1 == [problem.nodes[5]]
    assert best_move.segment2 == [problem.nodes[1]]

    # CASE 2: exchange nodes [1, 2] and node [4]
    route1 = [
        problem.nodes[3],
        problem.nodes[4],
    ]
    route2 = [
        problem.nodes[6],
        problem.nodes[5],
        problem.nodes[1],
        problem.nodes[2],
    ]
    solution = VRPSolution(problem)
    solution.add_route(route1)
    solution.add_route(route2)

    found_moves: list = search_cross_exchanges_from(
        solution=solution,
        cost_evaluator=evaluator,
        start_node=problem.nodes[1]
    )

    assert found_moves
    best_move = sorted(found_moves)[0]
    # assert best_move.improvement == 0
    assert best_move.segment1 == [problem.nodes[1], problem.nodes[2]]
    assert best_move.segment2 == [problem.nodes[4]]

    # CASE 3: exchange nodes [2, 1] and node [4]
    # Note that the segment [2, 1] is traversed reversely in the current route
    route1 = [
        problem.nodes[3],
        problem.nodes[4],
    ]
    route2 = [
        problem.nodes[6],
        problem.nodes[5],
        problem.nodes[1],
        problem.nodes[2],
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
    # assert best_move.improvement == 0
    assert best_move.segment1 == [problem.nodes[2], problem.nodes[1]]
    assert best_move.segment2 == [problem.nodes[4]]
    print(solution.routes)
    for route in solution.routes:
        for node in route.nodes:
            print(f'Node {node.node_id}, Time window: {node.time_window}, Arrival time: {node.arrival_time}')


def build_toy_problem_vrptw() -> tuple[VRPProblem, CostEvaluator]:
    # Based on toy.txt data
    depot = NodeWithTW(
        node_id=0,
        x_coordinate=50,
        y_coordinate=50,
        demand=0,
        is_depot=True,
        time_window=(0, 200),
        service_time=0
    )
    customers = [
        NodeWithTW(
            node_id=1,
            x_coordinate=35,
            y_coordinate=65,
            demand=10,
            is_depot=False,
            time_window=(45, 50),
            service_time=10
        ),
        NodeWithTW(
            node_id=2,
            x_coordinate=30,
            y_coordinate=55,
            demand=30,
            is_depot=False,
            time_window=(50, 80),
            service_time=10
        ),
        NodeWithTW(
            node_id=3,
            x_coordinate=40,
            y_coordinate=45,
            demand=10,
            is_depot=False,
            time_window=(0, 20),
            service_time=10
        ),
        NodeWithTW(
            node_id=4,
            x_coordinate=55,
            y_coordinate=70,
            demand=20,
            is_depot=False,
            time_window=(0, 100),
            service_time=10
        ),
        NodeWithTW(
            node_id=5,
            x_coordinate=75,
            y_coordinate=50,
            demand=10,
            is_depot=False,
            time_window=(50, 70),
            service_time=10
        ),
        NodeWithTW(
            node_id=6,
            x_coordinate=54,
            y_coordinate=35,
            demand=20,
            is_depot=False,
            time_window=(17, 20),
            service_time=10
        ),
    ]
    all_nodes = [depot] + customers
    vrp_problem = VRPProblem('VRPTW', all_nodes, 50, number_vehicles_required=3)
    vrp_evaluator = CostEvaluator(all_nodes, 50, {'capacity': 50, 'neighborhood_size': 5})

    return vrp_problem, vrp_evaluator


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

    vrp_problem = VRPProblem('CVRP', all_nodes, 3)
    vrp_evaluator = CostEvaluator(all_nodes, 3, {'neighborhood_size': 5})

    return vrp_problem, vrp_evaluator
