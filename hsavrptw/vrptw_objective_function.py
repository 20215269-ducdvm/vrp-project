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

from constraint_checker import VRPTWSequentialVectorConstraintChecker
from pyharmonysearch import ObjectiveFunctionInterface
from pyharmonysearch.harmony_search import harmony_search_serial, harmony_search


class VRPTWObjectiveFunctionSequentialVector(ObjectiveFunctionInterface):
    def __init__(self, arguments, problem_instance):
        self.problem_instance = problem_instance
        self.location_number = problem_instance['location_number']
        self.vehicle_number = problem_instance['vehicle_number']
        self._number_of_parameters = self.location_number + self.vehicle_number - 2
        self._discrete_values = []
        self._variable = []
        for i in range(self._number_of_parameters):
            self._discrete_values.append([i for i in range(0, self.location_number)])
            self._variable.append(True)

        # define all input parameters
        self._maximize = False  # minimize
        self._max_imp = int(arguments['--ni'])  # maximum number of improvisations
        self._hms = int(arguments['--hms'])  # harmony memory size
        self._hmcr = float(arguments['--hmcr'])  # harmony memory considering rate
        self._parmin = float(arguments['--parmin'])
        self._parmax = float(arguments['--parmax'])
        self._mpai = self.location_number - 1  # maximum pitch adjustment index

        # TODO check, if par is used directly or via function
        self._par = 0.5  # pitch adjusting rate

    def get_fitness(self, vector):
        """
        Calculate the fitness of the curr_solution vector
        :param vector: the curr_solution vector
        :return: fitness value if constraints are satisfied, otherwise return the degree of violation
        """
        constraints_value = self.get_constraints_value(vector)
        return self._get_total_distance(vector) if constraints_value == 0 else constraints_value

    def _get_total_distance(self, vector, depot=0):
        """
        Function to calculate total distance for this type of vector
        """
        total_distance = 0
        current_position = depot

        for node in vector:
            # calculate distance from current position to the next node
            total_distance += self.problem_instance['t'][current_position][node]

            # update current position
            current_position = node

            # if current note is depot, reset current position to depot
            # to start a new route
            if current_position == depot:
                current_position = depot

        if current_position != depot:
            total_distance += self.problem_instance['t'][current_position][depot]

        # ensure the final result is also properly truncated to exactly one decimal place
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

    def compute_par(self, gn):
        """Compute new pitch adjustment rate based on global number of improvisations """
        # "gn and NI represents the current iteration and the maximum number of iteration.
        # During the improvisation process, the value of PAR will be changed dynamically in descending number"
        self._par = self._parmax - gn * ((self._parmax - self._parmin) / self._max_imp)

    def get_constraints_value(self, vector):
        return VRPTWSequentialVectorConstraintChecker().set_params(self.problem_instance).get_constraints_value(vector)


from problem_parser import parse_problem, truncate_to_decimal
from docopt import docopt
from pyharmonysearch.harmony_search import HarmonySearchVRPTW

if __name__ == '__main__':
    arguments = docopt(__doc__)
    problem_instance = parse_problem(arguments['<problem_instance>'])
    obj_fun = VRPTWObjectiveFunctionSequentialVector(arguments, problem_instance)
    algorithm = HarmonySearchVRPTW(obj_fun)

    num_processes = cpu_count() - 1  # use number of logical CPUs - 1 so that I have one available for use
    num_iterations = 100

    results = (harmony_search(obj_fun, num_processes, num_iterations, algorithm))
    # results = (harmony_search_serial(obj_fun, num_iterations, algorithm))
    best_harmony = results.best_harmony
    print("Time elapsed:", results.elapsed_time)
    print("Best harmony:", best_harmony)
    print("Best fitness:", results.best_fitness)
    # print("Route:", VRPTWSequentialVectorConstraintChecker().set_params(problem_instance).convert_to_list_of_routes(best_harmony))

    # TEST minimal example
    # x7 = [3, 1, 2, 4, 0]  # [(0, [0, 3, 1, 2, 0])] 63.4
    # x8 = [3, 2, 4, 1, 5]  # [(0, [0, 3, 2, 0]), (1, [0, 1, 0])] 88.2
    # x9 = [1, 4, 2, 5, 3]  # [(0, [0, 1, 0]), (1, [0, 2, 0]), (2, [0, 3, 0])] 105.8
    # x10 = [5, 3, 1, 2, 4]  # [(1, [0, 0]), (0, [0, 3, 1, 2, 0])] 63.4
    # x11 = [4, 5, 3, 1, 2]  # [(0, [0, 0]), (1, [0, 3, 1, 2, 0])] 63.4
    # x12 = [1, 2, 3, 4, 5]  # [(0, [0, 1, 2, 3, 4, 0])] -70.19
    # print(obj_fun.get_fitness(x12))

    # TEST toy example
    # x1 = [4, 5, 7, 3, 1, 2, 8, 6]  # [(0, [0, 4, 5, 0]), (1, [0, 3, 1, 2, 0]), (2, [0, 6, 0])] 168.2
    # x2 = [3, 1, 2, 8, 6, 5, 7, 4]  # [(1, [0, 3, 1, 2, 0]), (0, [0, 6, 5, 0]), (2, [0, 4, 0])] 170.9
    # xb = [3, 1, 2, 7, 6, 5, 4, 8]  # [(0, [0, 3, 1, 2, 0]), (1, [0, 6, 5, 4, 0])] 153.5
    # x3 = [7, 6, 5, 1, 8, 3, 1, 2]  # [(0, [0, 0]), (1, [0, 6, 5, 4, 0]), (2, [0, 3, 1, 2, 0])] 153.5
    #
    # x_test1 = [7, 6, 1, 4, 4, 3, 3, 2]
    # print(obj_fun.get_fitness(x_test1))
