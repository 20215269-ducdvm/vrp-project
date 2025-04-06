
from kgls.local_search.operator_linkernighan import run_lin_kernighan_heuristic
from kgls.datastructure import Node, VRPProblem, CostEvaluator, VRPSolution


def build_problem() -> tuple[VRPProblem, CostEvaluator]:
    # all nodes in a line
    #  D  1  2  3  4
    depot = Node(node_id=0, x_coordinate=0, y_coordinate=0, demand=0, is_depot=True)
    customers = [
        Node(node_id=1, x_coordinate=10, y_coordinate=0, demand=1, is_depot=False),
        Node(node_id=2, x_coordinate=20, y_coordinate=0, demand=1, is_depot=False),
        Node(node_id=3, x_coordinate=30, y_coordinate=0, demand=1, is_depot=False),
        Node(node_id=4, x_coordinate=40, y_coordinate=0, demand=1, is_depot=False),
    ]
    all_nodes = [depot] + customers

    vrp_problem = VRPProblem(all_nodes, 5)
    vrp_evaluator = CostEvaluator(all_nodes, 5, {'neighborhood_size': 5})

    return vrp_problem, vrp_evaluator


def test_search_n_opt_moves():

    problem, evaluator = build_problem()
    # optimal route is 0-1-2-3-4-0 with route costs of 80

    # CASE 1: 2-opt
    # 0-2-1-3-4-0 with route costs 100
    # remove (0-2), (1-3) and add (0-1), (2-3)
    route = [
        problem.nodes[2],
        problem.nodes[1],
        problem.nodes[3],
        problem.nodes[4],
    ]
    solution = VRPSolution(problem)
    solution.add_route(route)

    run_lin_kernighan_heuristic(
        solution=solution,
        cost_evaluator=evaluator,
        route=solution.routes[0],
        max_depth=2
    )

    # There are multiple optimal solutions (like e.g., 0-1-3-4-2-0), so we just check the costs
    assert evaluator.get_solution_costs(solution) == 80


    # CASE 2: 3-opt
    # 0-3-1-2-4-0 with route costs of 120
    # remove (0-3), (2-4), (3-1) and add (3-2), (4-3), (1-0)
    route = [
        problem.nodes[3],
        problem.nodes[1],
        problem.nodes[2],
        problem.nodes[4],
    ]
    solution = VRPSolution(problem)
    solution.add_route(route)

    run_lin_kernighan_heuristic(
        solution=solution,
        cost_evaluator=evaluator,
        route=solution.routes[0],
        max_depth=3
    )

    # There are multiple optimal solutions (like e.g., 0-1-3-4-2-0), so we just check the costs
    assert evaluator.get_solution_costs(solution) == 80
