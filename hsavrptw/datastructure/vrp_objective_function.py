#!/usr/bin/python
"""hsa

Usage:
    hsa.py <_problem_instance> --hms=<hms> --hmcr=<hmcr> --par=<par> --ni=<ni>

Options:
    --hms=<hms>     Harmony memory size e.g. 10, 20, 30...
    --hmcr=<hmcr>   Harmony memory consideration rate e.g. 0.6, 0.7, 0.8
    --par=<par>     Pitch adjustment rate e.g. 0.9
    --ni=<ni>       Number of improvisations e.g. 500, 1000, 2000

"""
import random

from datastructure import VRPProblem, VRPSolution
from datastructure.route import compute_euclidean_distance
from helpers.helpers import multiply_and_floor, truncate_to_decimal
from pyharmonysearch import ObjectiveFunctionInterface


def has_duplicates(solution: list[int]) -> bool:
    """
    Check if a solution has duplicate customer IDs (excluding depot visits).

    Args:
        solution: List of node IDs representing a solution

    Returns:
        True if there are duplicate non-zero values, False otherwise
    """
    # Create a set to track seen customer IDs (non-zero values)
    seen = set()

    for node_id in solution:
        # Skip depot visits
        if node_id == 0:
            continue

        # If we've seen this customer before, it's a duplicate
        if node_id in seen:
            return True

        # Add this customer ID to seen set
        seen.add(node_id)

    # No duplicates found
    return False

def vrp_solution_to_vector(solution: VRPSolution) -> list[int]:
    """
    Convert a VRP solution back to a solution vector.

    Args:
        solution: The VRPSolution to convert

    Returns:
        A vector representation of the solution
    """
    vector = []

    for route in solution.routes:
        if route.size > 0:
            for node in route.customers:
                vector.append(node.node_id)
            vector.append(0)  # Add depot to mark end of route

    return vector


def vector_to_vrp_solution(vector, problem: VRPProblem) -> VRPSolution:
    """
    Convert a solution vector to a VRP solution.

    Args:
        vector: The solution vector to convert
        problem: The VRPProblem instance

    Returns:
        A VRPSolution instance
    """
    # This is a simplified implementation
    # You'll need to adapt this to your specific vector representation
    solution = VRPSolution(problem)

    # Parse the vector into routes
    current_route = []
    depot = 0  # Assuming depot is represented by 0

    for node_id in vector:
        if node_id == depot:
            if current_route:
                # Add completed route
                solution.add_route([problem.nodes[i] for i in current_route])
                current_route = []
        else:
            current_route.append(node_id)

    # Add any remaining nodes
    if current_route:
        solution.add_route([problem.nodes[i] for i in current_route])

    return solution

class VRPObjectiveFunction(ObjectiveFunctionInterface):
    def __init__(self, arguments, problem_instance: VRPProblem, number_of_parameters: int=None, initial_solution: list=None):
        self._problem_instance = problem_instance
        self._location_number = len(problem_instance.nodes)
        self._vehicle_number = problem_instance.number_vehicles_required

        self._number_of_parameters = self._location_number + self._vehicle_number - 2 if number_of_parameters is None else number_of_parameters

        self._discrete_values = []
        self._variable = []

        for i in range(self._number_of_parameters):
            self._discrete_values.append([i for i in range(self._location_number)])
            if initial_solution is not None:
                self._variable.append (True if initial_solution[i] != 0 else False) # we can't change the parameter value if it's 0, since it's the depot.
            else:
                self._variable.append(True)

        # define all input parameters
        self._maximize = False  # minimize
        self._max_imp = int(arguments['--ni'])  # maximum number of improvisations
        self._hms = int(arguments['--hms'])  # solution memory size
        self._hmcr = float(arguments['--hmcr'])  # solution memory considering rate
        self._mpai = self._location_number // 10  # maximum pitch adjustment index
        self._par = float(arguments['--par'])  # pitch adjusting rate

    @property
    def problem_instance(self):
        return self._problem_instance

    @property
    def location_number(self):
        return self._location_number

    def get_fitness(self, vector):
        """
        Calculate the fitness of the curr_solution vector
        :param vector: the curr_solution vector
        :return: fitness value if constraints are satisfied, otherwise return the degree of violation
        """
        violation_degree = self._get_constraints_value(vector)
        return self._get_total_distance(vector) if violation_degree == 0 else violation_degree * (-1)

    def _get_constraints_value(self, vector):
        """
        Calculate the degree of violation of the constraints
        :param vector: the curr_solution vector
        """
        violation_degree = 0

        if has_duplicates(vector):
            # Duplicate customers in the solution
            violation_degree += float('inf')

        violation_degree += self._capacity_violation(vector)

        if self._problem_instance.type == 'VRPTW':
            violation_degree += self._time_window_violation(vector)

        return violation_degree

    def _capacity_violation(self, vector):
        volume = 0
        capacity_violation = 0
        for i, node in enumerate(vector):
            if node == 0 or i == len(vector) - 1:
                if volume > self._problem_instance.capacity:
                    # Capacity violation
                    capacity_violation += (volume - self._problem_instance.capacity)
                # Reset volume when depot is reached
                volume = 0
            else:
                # Add the volume of the current node
                volume += self._problem_instance.nodes[node].demand
        return capacity_violation

    def _time_window_violation(self, vector):
        time_window_violation = 0
        prev_node_idx = 0
        self._problem_instance.nodes[prev_node_idx].arrival_time = 0
        for node in vector:
            prev_node = self._problem_instance.nodes[prev_node_idx]
            curr_node = self._problem_instance.nodes[node]

            curr_node.arrival_time = prev_node.arrival_time + prev_node.waiting_time + prev_node.service_time + compute_euclidean_distance(
                prev_node, curr_node)
            curr_node.waiting_time = max(0, multiply_and_floor(curr_node.time_window[0]) - curr_node.arrival_time)

            if curr_node.arrival_time > multiply_and_floor(curr_node.time_window[1]):
                time_window_violation += (curr_node.arrival_time - multiply_and_floor(curr_node.time_window[1]))

            if node == 0:
                # Reset arrival time and waiting time for depot
                curr_node.arrival_time = 0
                curr_node.waiting_time = 0

            prev_node_idx = node
        return truncate_to_decimal(time_window_violation)

    def _get_total_distance(self, vector):
        """
        Function to calculate total distance for this type of vector
        """
        total_distance = 0
        current_position = 0

        nodes = self._problem_instance.nodes

        for node in vector:
            # calculate distance from current position to the next node
            node1 = nodes[current_position]
            node2 = nodes[node]

            distance = compute_euclidean_distance(node1, node2)

            total_distance += distance

            # update current position
            current_position = node

            # if current note is depot, reset current position to depot
            # to start a new route
            if current_position == 0:
                current_position = 0

        if current_position != 0:
            final_node = nodes[current_position]
            depot = nodes[0]
            distance = compute_euclidean_distance(final_node, depot)

            total_distance += distance

        return truncate_to_decimal(total_distance)

    def get_value(self, i, j=None):
        return random.randrange(1, self._number_of_parameters) if j is None else self._discrete_values[i][j]

    def get_index(self, i, v):
        return self._discrete_values[i].index(v)

    def get_num_discrete_values(self, i):
        return len(self._discrete_values[i])

    def get_lower_bound(self, i):
        pass

    def get_upper_bound(self, i):
        pass

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        return True

    def get_num_parameters(self):
        return self._number_of_parameters

    def use_random_seed(self):
        pass

    def get_random_seed(self):
        pass

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        pass

    def maximize(self):
        return self._maximize

    def get_max_distance(self, node: int, neighborhood_size: int=100):
        # Generate a range of nodes based on the node index
        neighborhood = []

        # Add nodes in alternating order from both sides of the reference node
        left_idx = node - 1
        right_idx = node + 1

        while len(neighborhood) < neighborhood_size and (left_idx >= 1 or right_idx < self._location_number):
            # Add node to the right if it's valid
            if right_idx < self._location_number and len(neighborhood) < neighborhood_size:
                neighborhood.append(right_idx)
                right_idx += 1

            # Add node to the left if it's valid
            if left_idx >= 1 and len(neighborhood) < neighborhood_size:
                neighborhood.append(left_idx)
                left_idx -= 1

        # Return the maximum distance to the nodes in the neighborhood
        max_distance = 0
        for neighbor in neighborhood:
            distance = compute_euclidean_distance(self._problem_instance.nodes[node], self._problem_instance.nodes[neighbor])
            if distance > max_distance:
                max_distance = distance

        return max_distance