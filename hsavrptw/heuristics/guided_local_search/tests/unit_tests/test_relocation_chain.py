
from kgls.local_search.operator_relocation_chain import search_relocation_chains
from kgls.datastructure import Node, VRPProblem, CostEvaluator, VRPSolution


def build_problem() -> tuple[VRPProblem, CostEvaluator]:
    # Depot in the middle
    # 1   2   3
    #     D
    # 4   5   6
    depot = Node(node_id=0, x_coordinate=50, y_coordinate=20, demand=0, is_depot=True)
    customers = [
        Node(node_id=1, x_coordinate=0, y_coordinate=10, demand=1, is_depot=False),
        Node(node_id=2, x_coordinate=0, y_coordinate=20, demand=1, is_depot=False),
        Node(node_id=3, x_coordinate=0, y_coordinate=30, demand=1, is_depot=False),
        Node(node_id=4, x_coordinate=100, y_coordinate=10, demand=1, is_depot=False),
        Node(node_id=5, x_coordinate=100, y_coordinate=20, demand=1, is_depot=False),
        Node(node_id=6, x_coordinate=100, y_coordinate=20, demand=1, is_depot=False),
    ]
    all_nodes = [depot] + customers

    vrp_problem = VRPProblem(all_nodes, 3)
    vrp_evaluator = CostEvaluator(all_nodes, 3, {'neighborhood_size': 5})

    return vrp_problem, vrp_evaluator


def test_search_relocation_chains():

    problem, evaluator = build_problem()
    # optimal routes are D-1-2-3-D and D2-4-5-6-D

    # CASE 1: move node 4 to the start of route2 and then node 3 at the start of route1
    # (moving it after node 2 would be even better, but this is forbidden since node 4 was there previously)
    route1 = [
        problem.nodes[1],
        problem.nodes[2],
        problem.nodes[4],
    ]
    route2 = [
        problem.nodes[5],
        problem.nodes[3],
        problem.nodes[6],
    ]
    solution = VRPSolution(problem)
    solution.add_route(route1)
    solution.add_route(route2)

    relocation_chains: list = search_relocation_chains(
        solution=solution,
        cost_evaluator=evaluator,
        start_nodes=[problem.nodes[4]],
        max_depth=2
    )

    # Assertions
    # Check if valid moves were generated
    assert relocation_chains
    relocations = relocation_chains[0].relocations
    assert relocations[0].node_to_move == problem.nodes[4]
    assert relocations[0].move_after == problem.nodes[0]
    assert relocations[1].node_to_move == problem.nodes[3]
    assert relocations[1].move_after == problem.nodes[0]
    assert relocations[0].improvement == 90
    assert relocations[1].improvement == 180
