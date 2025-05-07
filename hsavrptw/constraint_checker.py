#!/usr/bin/python
"""hsa

Usage:
    hsa.py <_problem_instance>
"""
from abc import ABC, abstractmethod

from helpers.helpers import truncate_to_decimal
from problem_parser import parse_problem


class VRPTWAbstractConstraintChecker(ABC):
    def __init__(self):
        self.capacity = None  # total capacity
        self.n = None  # number of locations (depot + customers)
        self.v = None  # number of vehicles
        self.y = None  # 1 if customer is visited by vehicle k, 0 otherwise
        self.q = None  # demand of each customer
        self.a = None  # arrival time at each customer
        self.W = None  # Waiting time at each customer
        self.s = None  # service time at each customer
        self.t = None  # travel time between customers
        self.e = None  # start of time window for each customer
        self.l = None  # end of time window for each customer

    def set_params(self, problem_instance):
        # Initialize all common attributes from _problem_instance
        self.capacity = problem_instance['capacity']  # total capacity
        self.n = problem_instance['location_number']  # number of locations (depot + customers)
        self.v = problem_instance['vehicle_number']  # number of vehicles
        self.y = [[0 for _ in range(self.v)] for _ in
                  range(self.n)]  # 1 if customer is visited by vehicle k, 0 otherwise
        self.q = problem_instance['demand']  # demand of each customer
        self.a = [0] * self.n  # arrival time at each customer
        self.W = [0] * self.n  # Waiting time at each customer
        self.s = problem_instance['service_duration']  # service time at each customer
        self.t = problem_instance['t']  # travel time between customers
        self.e = problem_instance['time_window_start']  # start of time window for each customer
        self.l = problem_instance['time_window_end']  # end of time window for each customer
        return self

    def get_constraints_value(self, x) -> float:
        """
        Check if the curr_solution satisfies all constraints
        :param x: curr_solution to check
        :return: True if the curr_solution satisfies all constraints, False if violated basic VRP constraints (customer is visited by more than one vehicle, vehicle does not start from depot, customer is not served), degree of violation if violated time window or capacity constraints
        """
        violated_degree = 0
        print("checking vehicle number")
        if self.vehicle_number_exceed(x):
            return float('-inf')

        # Only one vehicle arriving at each customer and only one vehicle departing from each customer
        # This means that each customer is served by exactly one vehicle
        print("checking is served by one vehicle")
        if not self._is_served_by_one_vehicle(x):
            # print("Customer is served by more than one vehicle")
            return float('-inf')

        # keep track of the degree of capacity violation
        print("checking capacity")
        violated_degree += self._exceed_vehicle_capacity(x)

        # every vehicle starts from depot
        if not self.every_vehicle_starts_from_depot(x):
            # print("Vehicle does not start from depot")
            return float('-inf')

        # keep track of the degree of time window violation
        violated_degree += self._time_windows_violated(x)

        return violated_degree

    def _is_served_by_one_vehicle(self, x) -> bool:
        # Track the number of vehicles arriving at each customer
        arrived_customers = self.track_arrival(x)
        if arrived_customers is None:
            return False
        # Track the number of vehicles departing from each customer
        departed_customers = self.track_departure(x)
        if departed_customers is None:
            return False
        # If a vehicle arrives at a customer, it must also depart from that customer
        for k in range(self.v):
            for i in range(1, self.n):
                if arrived_customers[i][k] != departed_customers[i][k]:
                    # print("Arrived != Departed at customer", i)
                    return False

        self.y = arrived_customers

        # Each customer is served by exactly one vehicle
        for i in range(1, self.n):
            total_visited = 0
            for k in range(self.v):
                total_visited += self.y[i][k]
            if total_visited != 1:
                # print("Customer", i, "is visited by", total_visited, "vehicles")
                return False

        return True

    def _exceed_vehicle_capacity(self, x) -> int:
        capacity_violated = 0
        for k in range(self.v):
            total = 0
            for i in range(self.n):
                total = total + self.y[i][k] * self.q[i]
            capacity_violated += max(0, total - self.capacity)
        return -capacity_violated

    def every_vehicle_starts_from_depot(self, x) -> bool:
        for k in range(self.v):
            if self.y[0][k] != 1:
                return False
        return True

    @abstractmethod
    def vehicle_number_exceed(self, x) -> bool:
        """Check if the number of vehicles in the curr_solution exceeds the required number"""
        pass

    @abstractmethod
    def track_arrival(self, x) -> list[list[int]]:
        """Track the number of vehicles arriving at each customer"""
        pass

    @abstractmethod
    def track_departure(self, x) -> list[list[int]]:
        """Track the number of vehicles departing from each customer"""
        pass

    def _time_windows_violated(self, x) -> int:
        location_count = 0
        violation_degree = 0
        for k in range(self.v):
            current_location = 0
            while True:
                # find next location
                next_location = self.find_next_location(x, current_location, k)

                location_count += 1  # location_count at depot is duplicated v times since every vehicle starts from depot

                if next_location is None or next_location == 0:
                    break

                # calculate arrival time at next location
                self.a[next_location] = self.a[current_location] + self.W[current_location] + self.s[current_location] + \
                                        truncate_to_decimal(self.t[current_location][
                                                                next_location])

                # waiting time is either service start time - arriving time or 0
                # vehicle can arrive at a customer before time window, but it has to wait until the time window starts
                self.W[next_location] = max(self.e[next_location] - self.a[next_location], 0)

                # customer cannot be served after end of time window
                if self.a[next_location] > self.l[next_location]:
                    # print("Time window constraint violated at customer", next_location)
                    violation_degree += self.a[next_location] - self.l[next_location]
                current_location = next_location

        if location_count - self.v + 1 != self.n:
            # print("Not all customers are served")
            return float('-inf')

        return -violation_degree

    @abstractmethod
    def find_next_location(self, x, current_location, k) -> int | None:
        """
        Find next location according to how the curr_solution is encoded, the current location and the current vehicle
        :param x: curr_solution to check
        :param current_location: current location
        :param k: current vehicle
        :return: next location if exists, None otherwise
        """
        pass




class VRPTWNormalVectorConstraintChecker(VRPTWAbstractConstraintChecker):
    """
    Checker for normal representation of curr_solution
    """

    def track_arrival(self, x) -> list[list[int]] | None:
        arrived = [[0 for _ in range(self.v)] for _ in range(self.n)]
        for vehicle, route in enumerate(x):
            for i in range(len(route) - 1):
                arrived[route[i]][vehicle] += 1
        return arrived

    def track_departure(self, x) -> list[list[int]] | None:
        departed = [[0 for _ in range(self.v)] for _ in range(self.n)]
        for vehicle, route in enumerate(x):
            for i in range(len(route) - 1):
                departed[route[i]][vehicle] += 1
        return departed

    def find_next_location(self, x, current_location, k) -> int | None:
        for vehicle, route in enumerate(x):
            if vehicle == k:
                for i in range(len(route) - 1):
                    if route[i] == current_location:
                        return route[i + 1]
        return None

    def every_vehicle_starts_from_depot(self, x) -> bool:
        for vehicle, route in enumerate(x):
            if route[0] != 0:
                return False
        return True


def _check_for_duplicates(x):
    return len(x) == len(set(x))


def solution_to_routes(solution, depot=0) -> list[list[int]]:
    """
    Convert a flat curr_solution list to a list of routes.

    Args:
        solution (list): List representing the curr_solution with zeros as route delimiters
        depot (int, optional): The depot identifier. Defaults to 0.

    Returns:
        list: List of routes where each route starts and ends with the depot
    """
    routes = []
    current_route = [depot]  # start with depot

    for node in solution:
        if node == depot:
            # complete the current route by adding the depot
            current_route.append(depot)
            # add the completed route to routes if it contains nodes other than depot
            if len(current_route) > 2:  # More than just [depot, depot]
                routes.append(current_route)
            # start a new route
            current_route = [depot]
        else:
            # add node to current route
            current_route.append(node)

    # check if the last route is incomplete (doesn't end with depot)
    if len(current_route) > 1:  # Has at least one node besides depot
        current_route.append(depot)
        routes.append(current_route)

    return routes


class VRPTWSequentialVectorConstraintChecker(VRPTWAbstractConstraintChecker):
    """
    Checker for sequential representation of curr_solution
    Should go with the VRPTWObjectiveFunctionSequentialVector class.
    """
    def __init__(self):
        super().__init__()
        self.routes = None    
    def vehicle_number_exceed(self, x) -> bool:
        self.routes = solution_to_routes(x)
        return len(self.routes) > self.v

    def track_arrival(self, x) -> list[list[int]] | None:
        self.routes = solution_to_routes(x)
        if self.routes is None:
            return None
        arrived = [[0 for _ in range(self.v)] for _ in range(self.n)]
        for vehicle, route in enumerate(self.routes):
            for i in range(len(route) - 1):
                arrived[route[i]][vehicle] += 1
        return arrived

    def track_departure(self, x) -> list[list[int]] | None:
        departed = [[0 for _ in range(self.v)] for _ in range(self.n)]
        if self.routes is None:
            return None
        for vehicle, route in enumerate(self.routes):
            for i in range(len(route) - 1):
                departed[route[i]][vehicle] += 1
        return departed

    def find_next_location(self, x, current_location, k) -> int | None:
        self.routes = solution_to_routes(x)
        if self.routes is None:
            return None
        for vehicle, route in enumerate(self.routes):
            if vehicle == k:
                for i in range(len(route) - 1):
                    if route[i] == current_location:
                        return route[i + 1]
        return None

    def every_vehicle_starts_from_depot(self, x) -> bool:
        self.routes = solution_to_routes(x)
        for vehicle, route in enumerate(self.routes):
            if route[0] != 0:
                return False
        return True


if __name__ == '__main__':
    arguments = docopt(__doc__)
    problem_instance = parse_problem(arguments['<_problem_instance>'])

    x12 = [4, 5, 0, 3, 1, 2, 0, 6]  # [[0, 4, 5, 0], [0, 3, 1, 2, 0], [0, 6, 0]]
    x13 = [3, 1, 2, 0, 6, 5, 0, 4]  # [[0, 3, 1, 2, 0], [0, 6, 5, 0], [0, 4, 0]]
    xb = [3, 1, 2, 0, 6, 5, 4, 0]  # [[0, 3, 1, 2, 0], [0, 6, 5, 4, 0]] 153.5
    x1 = [3, 1, 6, 0, 2, 5, 0, 4]
    constraint_checker = VRPTWSequentialVectorConstraintChecker().set_params(problem_instance)

    print(solution_to_routes(x1), constraint_checker.get_constraints_value(x1))  # 0


