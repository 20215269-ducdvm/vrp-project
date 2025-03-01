from abc import ABC, abstractmethod
from typing import Any

from problem_parser import truncate_to_decimal


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
        # Initialize all common attributes from problem_instance
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

    def check_constraints(self, x):

        # Only one vehicle arriving at each customer and only one vehicle departing from each customer
        # This means that each customer is served by exactly one vehicle
        if not self._is_served_by_one_vehicle(x):
            # print("Customer is served by more than one vehicle")
            return False

        # vehicle capacity is not exceeded
        if not self._vehicle_capacity_not_exceeded(x):
            # print("Vehicle capacity exceeded")
            return False

        # every vehicle starts from depot
        if not self.every_vehicle_starts_from_depot(x):
            # print("Vehicle does not start from depot")
            return False

        # time window constraints are not violated
        if not self._time_window_constraints_not_violated(x):
            # print("Time window constraints violated")
            return False

        return True

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

    def _vehicle_capacity_not_exceeded(self, x) -> bool:
        for k in range(self.v):
            total = 0
            for i in range(self.n):
                total = total + self.y[i][k] * self.q[i]
            if total > self.capacity:
                return False
        return True

    def every_vehicle_starts_from_depot(self, x) -> bool:
        for k in range(self.v):
            if self.y[0][k] != 1:
                return False
        return True

    @abstractmethod
    def track_arrival(self, x) -> list[list[int]]:
        """Track the number of vehicles arriving at each customer"""
        pass

    @abstractmethod
    def track_departure(self, x) -> list[list[int]]:
        """Track the number of vehicles departing from each customer"""
        pass

    def _time_window_constraints_not_violated(self, x) -> bool:
        location_count = 0
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
                    return False
                current_location = next_location

        if location_count - self.v + 1 != self.n:
            # print("Not all customers are served")
            return False

        return True

    @abstractmethod
    def find_next_location(self, x, current_location, k) -> int | None:
        """Find next location according to how the solution is encoded, the current location and the current vehicle"""
        pass


class VRPTWBinaryVectorConstraintChecker(VRPTWAbstractConstraintChecker):
    """
    Checker for binary representation of solution
    Scrap this sht. Too many parameters, search space is too huge.
    """

    def track_arrival(self, x) -> list[list[int]]:
        arrived = [[0 for _ in range(self.v)] for _ in range(self.n)]

        for k in range(self.v):
            for j in range(self.n):
                for i in range(self.n):
                    arrived[j][k] += x[i][j][k]
        return arrived

    def track_departure(self, x) -> list[list[int]]:
        departed = [[0 for _ in range(self.v)] for _ in range(self.n)]

        for k in range(self.v):
            for i in range(self.n):
                for j in range(self.n):
                    departed[i][k] += x[i][j][k]
        return departed

    def find_next_location(self, x, current_location, k):
        for j in range(self.n):
            if x[current_location][j][k] == 1:
                return j
        return None


def _check_for_duplicates(x):
    return len(x) == len(set(x))


class VRPTWSequentialVectorConstraintChecker(VRPTWAbstractConstraintChecker):
    """
    Checker for sequential representation of solution
    Should go with the VRPTWObjectiveFunctionSequentialVector class.
    """

    # Valid cases: [1, 4, 2, 5, 3], [3, 2, 4, 1, 0], [1, 2, 3, 4, 0]
    # Invalid cases: [3, 2, 0, 1, 4], [4, 1, 2, 3, 5].
    # Rule: If there is a 0, we should immediately understand that there's no more vehicles to check,
    # any positive number comes after it completely invalidates the candidate solution.

    def __init__(self):
        super().__init__()
        self.routes = None

    def convert_to_list_of_routes(self, x) -> list[tuple[int | Any, list[Any]]] | None:
        if not _check_for_duplicates(x):
            return None
        routes = []  # list to store the visited customers by vehicle k
        temp_route = [0]
        vehicle_list = [i for i in range(self.v)]
        temp_vehicle_list = []
        idx = 0
        while True:
            if idx < len(x) - 1:
                if x[idx] == 0:
                    if x[idx + 1] != 0:
                        return None
                elif 0 < x[idx] <= self.n - 1:
                    temp_route.append(x[idx])
                elif x[idx] <= self.n + self.v - 2:
                    vehicle_number = x[idx] - self.n
                    temp_vehicle_list.append(vehicle_number)
                    temp_route.append(0)
                    routes.append((vehicle_number, temp_route))  # add the route to the list
                    temp_route = [0]  # reset the route
            else:
                if x[idx] == 0:
                    pass
                elif x[idx] <= self.n - 1:
                    temp_route.extend([x[idx], 0])
                    vehicle_number = next((v for v in vehicle_list if v not in temp_vehicle_list), None)
                    routes.append((vehicle_number, temp_route))
                else:
                    vehicle_number = x[idx] - self.n
                    temp_route.append(0)
                    temp_vehicle_list.append(vehicle_number)
                    routes.append((vehicle_number, temp_route))
                break
            idx += 1

        return routes

    def track_arrival(self, x) -> list[list[int]] | None:
        self.routes = self.convert_to_list_of_routes(x)
        if self.routes is None:
            return None
        arrived = [[0 for _ in range(self.v)] for _ in range(self.n)]
        for vehicle, route in self.routes:
            for i in range(len(route) - 1):
                arrived[route[i]][vehicle] += 1
        return arrived

    def track_departure(self, x) -> list[list[int]] | None:
        departed = [[0 for _ in range(self.v)] for _ in range(self.n)]
        if self.routes is None:
            return None
        for vehicle, route in self.routes:
            for i in range(len(route) - 1):
                departed[route[i]][vehicle] += 1
        return departed

    def find_next_location(self, x, current_location, k) -> int | None:
        self.routes = self.convert_to_list_of_routes(x)
        if self.routes is None:
            return None
        for vehicle, route in self.routes:
            if vehicle == k:
                for i in range(len(route) - 1):
                    if route[i] == current_location:
                        return route[i + 1]
        return None

    def every_vehicle_starts_from_depot(self, x) -> bool:
        self.routes = self.convert_to_list_of_routes(x)
        for vehicle, route in self.routes:
            if route[0] != 0:
                return False
        return True

if __name__ == '__main__':
    x1 = [1, 7, 2, 8, 3, 9, 4, 10, 5, 6]  # [(0, [0, 1, 0]), (1, [0, 2, 0]), (2, [0, 3, 0]), (3, [0, 4, 0]), (4, [0, 5, 6, 0])]
    x2 = [1, 2, 7, 3, 4, 8, 5, 9, 6, 10]  # [(0, [0, 1, 2, 0]), (1, [0, 3, 4, 0]), (2, [0, 5, 0]), (3, [0, 6, 0])]
    x3 = [1, 2, 7, 3, 4, 8, 5, 6, 9, 0]  # [(0, [0, 1, 2, 0]), (1, [0, 3, 4, 0]), (2, [0, 5, 6, 0])]
    x4 = [1, 2, 3, 4, 7, 5, 6, 8, 0, 0]  # [(0, [0, 1, 2, 3, 4, 0]), (1, [0, 5, 6, 0])]
    x5 = [1, 0, 2, 4, 7, 5, 6, 8, 0, 0] # None
    x6 = [1, 2, 7, 3, 8, 4, 9, 5, 6, 10] # [(0, [0, 1, 2, 0]), (1, [0, 3, 0]), (2, [0, 4, 0]), (3, [0, 5, 6, 0])]

    x7 = [3, 1, 2, 4, 0] # [(0, [0, 3, 1, 2, 0])]

    x8 = [3, 2, 4, 1, 5] # [(0, [0, 3, 2, 0]), (1, [0, 1, 0])]

    x9 = [1, 4, 2, 5, 3] # [(0, [0, 1, 0]), (1, [0, 2, 0]), (2, [0, 3, 0])]

    x10 = [5, 3, 1, 2, 4]
    x11 = [4, 5, 3, 1, 2]

    x12 = [4, 5, 7, 3, 1, 2, 8, 6] # [(0, [0, 4, 5, 0]), (1, [0, 3, 1, 2, 0]), (2, [0, 6, 0])] 152.7
    x13 = [3, 1, 2, 8, 6, 5, 7, 4] # [(1, [0, 3, 1, 2, 0]), (0, [0, 6, 5, 0]), (2, [0, 4, 0])] 150.29999999999998
    xb = [3, 1, 2, 7, 6, 5, 4, 8] # [(0, [0, 3, 1, 2, 0]), (1, [0, 6, 5, 4, 0])]

    constraint_checker = VRPTWSequentialVectorConstraintChecker()
    constraint_checker.v = 3
    constraint_checker.n = 7

    print(constraint_checker.convert_to_list_of_routes(xb))
