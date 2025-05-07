from .node import Node


class VRPProblem:
    # TODO make immutable
    type: str = None
    number_vehicles_required: int = None
    nodes: list[Node]
    capacity: int
    bks: float

    def __init__(self, instance_type: str, nodes: list[Node], capacity: int, bks: float = float('inf'), number_vehicles_required: int = None):
        assert instance_type in ['CVRP', 'VRPTW'], f"Unknown instance type: {instance_type}. Expected 'CVRP' or 'VRPTW'."
        self.type: str = instance_type
        self.nodes: list[Node] = nodes
        self.capacity: int = capacity
        self.bks: float = bks
        self.number_vehicles_required: int = number_vehicles_required
        self.customers = [node for node in nodes if not node.is_depot]
        self.depot = [node for node in nodes if node.is_depot][0]
