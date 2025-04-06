from kgls.datastructure import Node, Edge, Route, VRPProblem, VRPSolution, CostEvaluator


def test_compute_edge_width_perpendicular():
    depot = Node(0, 10, 10, 0, True)
    customers = [
        Node(1, 0, 0, 1, False),
        Node(2, 0, 20, 1, False)
    ]
    nodes = [depot] + customers

    evaluator = CostEvaluator(nodes, 4, {'neighborhood_size': 5})
    center_x, center_y = evaluator._compute_route_center(nodes)

    width = evaluator._compute_edge_width(
        Edge(customers[0], customers[1]),
        center_x,
        center_y,
        depot
    )

    assert width == 20.0


def test_compute_edge_width_line():
    depot = Node(0, 10, 10, 0, True)
    customers = [
        Node(1, 20, 10, 1, False),
        Node(2, 30, 10, 1, False)
    ]
    nodes = [depot] + customers

    evaluator = CostEvaluator(nodes, 4, {'neighborhood_size': 5})
    center_x, center_y = evaluator._compute_route_center(nodes)

    width = evaluator._compute_edge_width(
        Edge(customers[0], customers[1]),
        center_x,
        center_y,
        depot
    )

    assert width == 0.0


def test_determine_edge_badness():
    depot = Node(0, 0, 0, 0, True)
    customers = [
        Node(1, 10, 0, 1, False),
        Node(2, 30, 0, 1, False),
        Node(3, 60, 0, 1, False),
    ]
    nodes = [depot] + customers

    problem = VRPProblem(nodes, 3)
    evaluator = CostEvaluator(nodes, 5, {'neighborhood_size': 5})

    solution = VRPSolution(problem)
    solution.add_route(customers)

    # width as criterium activated
    evaluator.determine_edge_badness(solution.routes)
    # length as criterium activated
    evaluator.determine_edge_badness(solution.routes)

    edge = evaluator.get_and_penalize_worst_edge()
    assert edge == Edge(nodes[3], nodes[0])
    assert edge.value == 30

    edge = evaluator.get_and_penalize_worst_edge()
    assert edge == Edge(nodes[2], nodes[3])
    assert edge.value == 15

    edge = evaluator.get_and_penalize_worst_edge()
    assert edge == Edge(nodes[3], nodes[0])
    assert edge.value == 20

    edge = evaluator.get_and_penalize_worst_edge()
    assert edge == Edge(nodes[1], nodes[2])
    assert edge.value == 10