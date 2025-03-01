#!/usr/bin/python
"""hsa

Usage:
    hsa.py <problem_instance> --hms=<hms> --hmcr=<hmcr> --parmax=<par> --parmin=<parmin> --ni=<ni>

Options:
    --hms=<hms>     Harmony memory size e.g. 10, 20, 30...
    --hmcr=<hmcr>   Harmony memory consideration rate e.g. 0.6, 0.7, 0.8
    --ni=<ni>       Number of improvisations e.g. 500, 1000, 2000
    --parxmax=<parmax>  Maximal pitch adjustment rate e.g. 0.9
    --parxmin=<parmin>  Minimal pitch adjustment rate e.g. 0.3

"""
import random
from multiprocessing import cpu_count

from constraint_checker import VRPTWBinaryVectorConstraintChecker, VRPTWSequentialVectorConstraintChecker
from pyharmonysearch import ObjectiveFunctionInterface
from pyharmonysearch.harmony_search import harmony_search_serial, harmony_search


class VRPTWObjectiveFunctionBinaryVector(ObjectiveFunctionInterface):
    """
    For this class, the solution is a vector consisting of a list of numbers representing the routes for all vehicles
    it is converted to a 3D matrix x[i][j][k] where i is the starting city, j is the ending city, and k is the vehicle number
    x[i][j][k] = 1 if vehicle k traveled from i to j
    x[i][j][k] = 0 otherwise
    However the number of parameters are too large, and the search space is 2^(number of parameters), which leaves too many
    redundant solutions. Therefore, we need to use a different encoding to reduce the number of parameters.
    """

    def __init__(self, arguments, problem_instance):
        self.problem_instance = problem_instance
        self.location_number = problem_instance['location_number']
        self.vehicle_number = problem_instance['vehicle_number']
        # x[i][j][k] = 1 iff vehicle k traveled from i to j
        # 0 otherwise
        self._number_of_parameters = self.location_number ** 2 \
                                     * self.vehicle_number
        self._discrete_values = []
        self._variable = []
        for i in range(self._number_of_parameters):
            self._discrete_values.append([0, 1])
            (from_city, to_city, vehicle_number) = self.index_to_ijk(i)
            if from_city == to_city:
                self._variable.append(False)  # vehicle can't travel from city to itself
            else:
                self._variable.append(True)

        # define all input parameters
        self._maximize = False  # minimize
        self._max_imp = int(arguments['--ni'])  # maximum number of improvisations
        self._hms = int(arguments['--hms'])  # harmony memory size
        self._hmcr = float(arguments['--hmcr'])  # harmony memory considering rate
        self._parmin = float(arguments['--parmin'])
        self._parmax = float(arguments['--parmax'])
        self._mpai = 1

        # TODO check, if par is used directly or via function
        self._par = 0.5  # pitch adjusting rate

    def ijk_to_index(self, i, j, k):
        index = i * self.vehicle_number * self.location_number + j * self.vehicle_number + k
        return index

    def index_to_ijk(self, index):
        i = index // (self.vehicle_number * self.location_number)
        j = (index % (self.vehicle_number * self.location_number)) // self.vehicle_number
        k = index % self.vehicle_number
        return i, j, k

    def make_x_from_vector(self, vector):
        x = [[[0 for k in range(self.vehicle_number)] for j in range(self.location_number)] for i in
             range(self.location_number)]
        for i in range(self.location_number):
            for j in range(self.location_number):
                for k in range(self.vehicle_number):
                    x[i][j][k] = vector[self.ijk_to_index(i, j, k)]
        return x

    def _get_total_distance(self, x):
        total_distance = 0.0
        for k in range(self.vehicle_number):
            for i in range(self.location_number):
                for j in range(self.location_number):
                    total_distance += x[i][j][k] * self.problem_instance['t'][i][j]
        return total_distance

    def get_fitness(self, vector):
        x = self.make_x_from_vector(vector)

        return self._get_total_distance(x)

    def get_value(self, i, j=None):
        return random.randrange(2) if j is None else self._discrete_values[i][j]
        # return self._discrete_values[i][j]

    def get_num_discrete_values(self, i):
        # there will be always 0 or 1
        # return 2
        return len(self._discrete_values[i])

    def get_index(self, i, v):
        # in [1, 2] index of 1 is 0 and index of 2 is 1
        # return v
        return self._discrete_values[i].index(v)

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        # All variables are discrete
        return True

    def get_num_parameters(self):
        # compute number of parameters
        # return len(self._discrete_values)
        return self._number_of_parameters

    def use_random_seed(self):
        # What ever that means :D
        return hasattr(self, '_random_seed') and self._random_seed

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def set_par(self, gn):
        """Compute new pitch adjustment rate based on global number of improvisations """
        # "gn and NI represents the current iteration and the maximum number of iteration.
        # During the improvisation process, the value of PAR will be changed dynamically in descending number"
        self._par = self._parmax - gn * ((self._parmax - self._parmin) / self._max_imp)

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        # TODO remove, when it runs
        return 0.5

    def maximize(self):
        return self._maximize

    def check_constraints(self, vector):
        x = self.make_x_from_vector(vector)
        return VRPTWBinaryVectorConstraintChecker().set_params(problem_instance).check_constraints(x)


class VRPTWObjectiveFunctionSequentialVector(ObjectiveFunctionInterface):
    """
    For this class, x is a vector consisting of a list of numbers representing the routes for all vehicles
    Can be assigned different discrete values, including the encoding of vehicles
    For example, for a problem requiring 3 vehicles and 3 customers, [1, 4, 2, 5, 3] means that
    the route for vehicle 1 is 0 -> 1 -> 0, for vehicle 2 is 0 -> 2 -> 0, for vehicle 3 is 0 -> 3 -> 0
    4 represents vehicle 1, 5 represents vehicle 2.

    In case where we don't need all vehicles, we can assign a value to represent that the vehicle is not used
    For example, problem requires 3 vehicles and 3 customers, but the solution only needs 2 vehicles to serve all customers
    The route for vehicle 1 is 0 -> 3 -> 2 -> 0, for vehicle 2 is 0 -> 1 -> 0, then the appropriate encoding is
    [3, 2, 4, 1, 5]

    In case where only one vehicle is needed, for example vehicle 1's route is 0 -> 1 -> 2 -> 3 -> 0, then the encoding is
    [1, 2, 3, 0, 0]

    We can create a new solution vector by changing the route between vehicles.
    For example, [3, 2, 4, 1, 0] is equivalent to [1, 4, 3, 2, 0]
    since both mean that one vehicle serves customer 3 and 2, and the other serves customer 1.
    
    In general: with v vehicles and n customers, and m is the number of required vehicles.
    Total number of parameters to encode the solution: n + v - 1    
    How many parameters are required for the solution?
    - n parameters for n solutions
    - m required vehicles → how many parameters for encoding?
        - If m = v, we need m - 1 parameters
        - If m < v, we still need m parameters to encode the vehicles
            - If m  <= v - 2, number of parameters is not enough, we need to add 0’s at the end.
    """

    def __init__(self, arguments, problem_instance):
        self.problem_instance = problem_instance
        self.location_number = problem_instance['location_number']
        self.vehicle_number = problem_instance['vehicle_number']
        self._number_of_parameters = self.location_number + self.vehicle_number - 2
        self._discrete_values = []
        self._variable = []
        for i in range(self._number_of_parameters):
            self._discrete_values.append([i for i in range(0, self._number_of_parameters + 1)])
            self._variable.append(True)

        # define all input parameters
        self._maximize = False  # minimize
        self._max_imp = int(arguments['--ni'])  # maximum number of improvisations
        self._hms = int(arguments['--hms'])  # harmony memory size
        self._hmcr = float(arguments['--hmcr'])  # harmony memory considering rate
        self._parmin = float(arguments['--parmin'])
        self._parmax = float(arguments['--parmax'])
        self._mpai = 1

        # TODO check, if par is used directly or via function
        self._par = 0.5  # pitch adjusting rate

    def get_fitness(self, vector):
        total_distance = 0
        for i in range(len(vector)):
            if vector[i] == 0:
                break
            if i == 0 or vector[i - 1] >= self.location_number:
                if vector[i] >= self.location_number:
                    continue
                total_distance += self.problem_instance['t'][0][vector[i]]
                if i == len(vector) - 1:
                    total_distance += self.problem_instance['t'][vector[i]][0]
            elif vector[i] >= self.location_number:
                total_distance += self.problem_instance['t'][vector[i - 1]][0]
            else:
                total_distance += self.problem_instance['t'][vector[i - 1]][vector[i]]
                if i == len(vector) - 1:
                    total_distance += self.problem_instance['t'][vector[i]][0]

        # Ensure the final result is also properly truncated to exactly one decimal place
        return truncate_to_decimal(total_distance)

    def get_value(self, i, j=None):
        return random.randrange(1, self._number_of_parameters) if j is None else self._discrete_values[i][j]

    def get_index(self, i, v):
        return self._discrete_values[i].index(v)

    def get_num_discrete_values(self, i):
        return len(self._discrete_values[i])

    def get_lower_bound(self, i):
        # only used for continuous variables, so I'll pass
        pass

    def get_upper_bound(self, i):
        # only used for continuous variables, so I'll pass
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

    def check_constraints(self, vector):
        return VRPTWSequentialVectorConstraintChecker().set_params(self.problem_instance).check_constraints(vector)


from problem_parser import parse_problem, truncate_to_decimal
from docopt import docopt
from pyharmonysearch.harmony_search import HarmonySearchWithConstraints

if __name__ == '__main__':
    arguments = docopt(__doc__)
    problem_instance = parse_problem(arguments['<problem_instance>'])
    obj_fun = VRPTWObjectiveFunctionSequentialVector(arguments, problem_instance)
    algorithm = HarmonySearchWithConstraints(obj_fun)

    num_processes = cpu_count() - 1  # use number of logical CPUs - 1 so that I have one available for use
    num_iterations = 100

    results = (harmony_search(obj_fun, num_processes, num_iterations, algorithm))
    # results = (harmony_search_serial(obj_fun, num_iterations, algorithm))
    best_harmony = results.best_harmony
    print("Time elapsed:", results.elapsed_time)
    print("Best harmony:", best_harmony)
    print("Best fitness:", results.best_fitness)
    print("Route:", VRPTWSequentialVectorConstraintChecker().set_params(problem_instance).convert_to_list_of_routes(best_harmony))

    # x1 = [4, 5, 7, 3, 1, 2, 8, 6]  # [(0, [0, 4, 5, 0]), (1, [0, 3, 1, 2, 0]), (2, [0, 6, 0])] 168.2
    # x2 = [3, 1, 2, 8, 6, 5, 7, 4]  # [(1, [0, 3, 1, 2, 0]), (0, [0, 6, 5, 0]), (2, [0, 4, 0])] 170.9
    # xb = [3, 1, 2, 7, 6, 5, 4, 8]  # [(0, [0, 3, 1, 2, 0]), (1, [0, 6, 5, 4, 0])] 153.5
    # x3 = [7, 6, 5, 4, 8, 3, 1, 2]  # [(0, [0, 0]), (1, [0, 6, 5, 4, 0]), (2, [0, 3, 1, 2, 0])] 153.5
    #
    # print(obj_fun.get_fitness(x3))
