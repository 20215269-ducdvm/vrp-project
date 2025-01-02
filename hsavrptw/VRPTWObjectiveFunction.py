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

from pyharmonysearch import ObjectiveFunctionInterface, harmony_search
from pyharmonysearch.harmony_search import harmony_search_serial

class VRPTWObjectiveFunction(ObjectiveFunctionInterface):
    def __init__(self, arguments, problem_instance):
        self.problem_instance = problem_instance
        self.customer_number = problem_instance['customer_number']
        self.vehicle_number = problem_instance['vehicle_number']
        # x[i][j][k] = 1 iff vehicle k traveled from i to j
        # 0 otherwise
        self.number_of_parameters = self.customer_number ** 2 \
                                    * self.vehicle_number
        self._discrete_values = []
        self._variable = []
        for i in range(self.number_of_parameters):
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
        self._fitness_log = []

    def ijk_to_index(self, i, j, k):
        index = i * self.vehicle_number * self.customer_number + j * self.vehicle_number + k
        return index

    def index_to_ijk(self, index):
        i = index // (self.vehicle_number * self.customer_number)
        j = (index % (self.vehicle_number * self.customer_number)) // self.vehicle_number
        k = index % self.vehicle_number
        return i, j, k

    def make_x_from_vector(self, vector):
        x = [[[0 for k in range(self.vehicle_number)] for j in range(self.customer_number)] for i in
             range(self.customer_number)]
        for i in range(self.customer_number):
            for j in range(self.customer_number):
                for k in range(self.vehicle_number):
                    x[i][j][k] = vector[self.ijk_to_index(i, j, k)]
        return x

    def _get_total_distance(self, x):
        total_distance = 0.0
        for k in range(self.vehicle_number):
            for i in range(self.customer_number):
                for j in range(self.customer_number):
                    if x[i][j][k] == 1:
                        total_distance += self.problem_instance['t'][i][j]
        return total_distance

    def _get_max_time(self, x):
        max_time = 0.0
        for k in range(self.vehicle_number):
            vehicle_time = 0.0
            for i in range(self.customer_number):
                for j in range(self.customer_number):
                    if x[i][j][k] == 1:
                        vehicle_time += self.problem_instance['t'][i][j]
            if vehicle_time > max_time:
                max_time = vehicle_time
        return max_time

    def get_fitness(self, vector):
        x = self.make_x_from_vector(vector)

        # check constraints
        if not check_constraints(x, self.problem_instance):
            return float("inf")

        return self._get_max_time(x)

    def get_value(self, i, j=None):
        return random.randrange(2) if j is None else self._discrete_values[i][j]
        # return self._discrete_values[i][j]

    def get_num_discrete_values(self, i):
        # there will be always 0 or 1
        # return 2
        return len(self._discrete_values[i])

    def get_index(self, i, v):
        # index of 0 is 0 and index of 1 is 1 in [0, 1]
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
        return self.number_of_parameters

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

    def get_fitness_log(self):
        return frozenset(self._fitness_log)

    def add_fitness_log(self, fitness):
        self._fitness_log.append(fitness)

from problemParser import parse_problem, check_constraints
from docopt import docopt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    arguments = docopt(__doc__)
    problem_instance = parse_problem(arguments['<problem_instance>'])
    obj_fun = VRPTWObjectiveFunction(arguments, problem_instance)
    num_processes = cpu_count() - 1  # use number of logical CPUs - 1 so that I have one available for use
    num_iterations = 100
    # results = (harmony_search(obj_fun, num_processes, num_iterations))
    # results = (harmony_search_serial(obj_fun, num_iterations))
    # best_harmony = results.best_harmony

    # Plot the fitness values
    # print(list(obj_fun.get_fitness_log()))
    # plt.plot(obj_fun.get_fitness_log())
    # plt.xlabel('Iteration')
    # plt.ylabel('Fitness')
    # plt.title('Fitness over Iterations')
    # plt.show()

    # best_harmony =  [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0] # 1 vehicle
    # best_harmony = [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #                 0]  # 2 vehicles

    # ==========================3 VEHICLES=====================================
    # best_harmony = [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    #                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # best_harmony = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    #                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    best_harmony = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    x = obj_fun.make_x_from_vector(best_harmony)
    print("Check constraints: ", check_constraints(x, problem_instance))
    # print("Elapsed time: ", results.elapsed_time)
    print("Best harmony: ", best_harmony)
    print("Vector: ", x)
    # print("Best fitness:", results.best_fitness)
    print("Best fitness:", obj_fun.get_fitness(best_harmony))
