from kgls.datastructure import Node, VRPProblem, Route, VRPSolution


def test_insert_nodes_after():
    depot = Node(0, 0, 0, 0, True)
    customers = [
        Node(1, 0, 0, 1, False),
        Node(2, 0, 0, 1, False),
        Node(3, 0, 0, 1, False),
        Node(4, 0, 0, 1, False),
    ]
    nodes = [depot] + customers
    problem = VRPProblem(nodes, 5)

    solution = VRPSolution(problem)
    solution.add_route(customers[:2])

    solution.insert_nodes_after(
        nodes_to_be_inserted=customers[2:],
        move_after_node=customers[0],
        route=solution.routes[0]
    )

    solution.routes[0].validate()
    assert solution.routes[0].print() == "0-1-3-4-2-0"


def test_remove_nodes():
    depot = Node(0, 0, 0, 0, True)
    customers = [
        Node(1, 0, 0, 1, False),
        Node(2, 0, 0, 1, False),
        Node(3, 0, 0, 1, False),
        Node(4, 0, 0, 1, False),
    ]
    nodes = [depot] + customers
    problem = VRPProblem(nodes, 5)

    solution = VRPSolution(problem)
    solution.add_route(customers)

    solution.remove_nodes(customers[1:3][::-1])

    solution.routes[0].validate()
    assert solution.routes[0].print() == "0-1-4-0"

    solution.remove_nodes([customers[0]])

    solution.routes[0].validate()
    assert solution.routes[0].print() == "0-4-0"
