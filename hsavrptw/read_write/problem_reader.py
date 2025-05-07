import os

from datastructure import VRPProblem
from datastructure.node import Node, NodeWithTW



def read_vrp_instance(file_path: str) -> VRPProblem:
    """
    Unified reader for both CVRP and VRPTW problem instances.
    Automatically detects file format and parses accordingly.

    Args:
        file_path: Path to the problem instance file

    Returns:
        VRPProblem instance with appropriate node types
    """
    # Determine file type based on extension or content
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        file.seek(0)  # Reset file pointer

        # Check if it's a CVRP instance convention
        if first_line.startswith('N'):
            return read_cvrp_instance(file_path)
        # Otherwise assume it's a VRPTW instance convention
        else:
            return read_vrptw_instance(file_path)


def read_cvrp_instance(file_path: str) -> VRPProblem:
    """Read CVRP instance files (like X-n101-k25.vrp)"""
    nodes = dict()
    capacity: int = 0

    with open(file_path, 'r') as file:
        current_section = None
        for line in file:
            line = line.strip()

            if line == '':
                continue

            if line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1].strip())

            elif not line[0].isdigit():
                current_section = line
                continue

            elif current_section == "NODE_COORD_SECTION":
                parts = line.split()
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                nodes[node_id] = {
                    'id': node_id,
                    'x': x,
                    'y': y
                }

            elif current_section == "DEMAND_SECTION":
                parts = line.split()
                node_id = int(parts[0].strip())
                demand = int(parts[1].strip())
                nodes[node_id].update({
                    'demand': demand
                })

            elif line == "EOF":
                break

        # Try to read the best known solution
        sol_file_path = file_path.replace('.vrp', '.sol')
        best_solution = read_best_known_solution(sol_file_path) if os.path.exists(sol_file_path) else float('inf')

    vrp_nodes = [
        Node(
            node_id=node['id'] - 1, # convert to 0-based index
            x_coordinate=node['x'],
            y_coordinate=node['y'],
            demand=node['demand'],
            is_depot=node['demand'] == 0,
        )
        for node in nodes.values()
    ]
    return VRPProblem(instance_type='CVRP', nodes=vrp_nodes, capacity=capacity, bks=best_solution)


def read_vrptw_instance(file_path: str) -> VRPProblem:
    """Read VRPTW (Solomon) instance files (like C101.txt)"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    problem_name = lines[0].strip()
    vehicle_info = [int(item) for item in lines[4].split()]
    vehicle_number, capacity = vehicle_info

    customer_lines = [line for line in lines[9:] if line.strip() and line.split()]

    nodes = []
    for line in customer_lines:
        parts = [int(float(item)) for item in line.split()]
        if len(parts) >= 7:
            [number, x_coord, y_coord, demand, ready_time, due_time, service_time] = parts[:7]

            # Create NodeWithTW for each customer
            node = NodeWithTW(
                node_id=number,
                x_coordinate=x_coord,
                y_coordinate=y_coord,
                demand=demand,
                is_depot=(number == 0),  # Depot is typically node 0
                time_window=(ready_time, due_time),
                service_time=service_time,
                arrival_time=0,
                waiting_time=0
            )
            nodes.append(node)

    # Try to find BKS from literature (not stored in file for Solomon instances)
    # Could be extended to look for a matching .sol file or use a mapping of known BKS values
    best_solution = float('inf')

    return VRPProblem(instance_type='VRPTW', nodes=nodes, capacity=capacity, bks=best_solution,
                      number_vehicles_required=vehicle_number)


def read_best_known_solution(file_path: str) -> float:
    """Read best known solution from a .sol file"""
    cost = float('inf')
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('Cost'):
                    parts = line.split()
                    cost = float(parts[1].strip())
    except FileNotFoundError:
        pass
    return cost
