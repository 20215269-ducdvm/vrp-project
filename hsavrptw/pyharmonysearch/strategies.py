# import copy
# import random
# from abc import ABC, abstractmethod
#
# TODO: Make code pretty later
#
# class InitializationStrategy(ABC):
#     def __init__(self, objective_function):
#         self._obj_fun = objective_function
#         self._harmony_memory = list()
#         self._harmony_history = list()
#     @abstractmethod
#     def initialize(self, initial_harmonies=None):
#         pass
#
#
# class InitializationVanilla(InitializationStrategy):
#     def initialize(self, initial_harmonies=None):
#         """
#             Initialize harmony_memory, the matrix (list of lists) containing the various harmonies (curr_solution vectors). Note
#             that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
#             merely stores previous harmonies.
#
#             If harmonies are provided, then use them instead of randomly initializing them.
#
#             Populate harmony_history with initial harmony memory.
#         """
#         if initial_harmonies is not None:
#             # verify that the initial harmonies are provided correctly
#
#             if len(initial_harmonies) != self._obj_fun.get_hms():
#                 raise ValueError('Number of initial harmonies does not equal to the harmony memory size.')
#
#             num_parameters = self._obj_fun.get_num_parameters()
#             for i in range(len(initial_harmonies)):
#                 num_parameters_initial_harmonies = len(initial_harmonies[i])
#                 if num_parameters_initial_harmonies != num_parameters:
#                     raise ValueError('Number of parameters in initial harmonies does not match that defined.')
#         else:
#             initial_harmonies = list()
#             for i in range(0, self._obj_fun.get_hms()):
#                 harmony = list()
#                 for j in range(0, self._obj_fun.get_num_parameters()):
#                     self._random_selection(harmony, j)
#                 initial_harmonies.append(harmony)
#
#         for i in range(0, self._obj_fun.get_hms()):
#             fitness = self._obj_fun.get_fitness(initial_harmonies[i])
#             self._harmony_memory.append((initial_harmonies[i], fitness))
#
#         harmony_list = {'gen': 0, 'harmonies': copy.deepcopy(self._harmony_memory)}
#         self._harmony_history.append(harmony_list)
#
#
# class PitchAdjustmentStrategy(ABC):
#     @abstractmethod
#     def pitch_adjustment(self, harmony, i):
#         pass
#
# class PitchAdjustmentVanilla(PitchAdjustmentStrategy):
#     def _pitch_adjustment(self, harmony, i):
#         """
#             If variable, randomly adjust the pitch up or down by some amount. This is the only place in the algorithm where there
#             is an explicit difference between continuous and discrete variables.
#
#             The probability of adjusting the pitch either up or down is fixed at 0.5. The maximum pitch adjustment proportion (mpap)
#             and maximum pitch adjustment index (mpai) determine the maximum amount the pitch may change for continuous and discrete
#             variables, respectively.
#
#             For example, suppose that it is decided via coin flip that the pitch will be adjusted down. Also suppose that mpap is set to 0.25.
#             This means that the maximum value the pitch can be dropped will be 25% of the difference between the lower bound and the current
#             pitch. mpai functions similarly, only it relies on indices of the possible values instead.
#         """
#         if self._obj_fun.is_variable(i):
#             if self._obj_fun.is_discrete(i):
#                 current_index = self._obj_fun.get_index(i, harmony[i])
#                 # discrete variable
#                 if random.random() < 0.5:
#                     # adjust pitch down
#                     harmony[i] = self._obj_fun.get_value(i, current_index - random.randint(0,
#                                                                                            min(self._obj_fun.get_mpai(),
#                                                                                                current_index)))
#                 else:
#                     # adjust pitch up
#                     harmony[i] = self._obj_fun.get_value(i, current_index + random.randint(0,
#                                                                                            min(self._obj_fun.get_mpai(),
#                                                                                                self._obj_fun.get_num_discrete_values(
#                                                                                                    i) - current_index - 1)))
#             else:
#                 # continuous variable
#                 if random.random() < 0.5:
#                     # adjust pitch down
#                     harmony[i] -= (harmony[i] - self._obj_fun.get_lower_bound(
#                         i)) * random.random() * self._obj_fun.get_mpap()
#                 else:
#                     # adjust pitch up
#                     harmony[i] += (self._obj_fun.get_upper_bound(i) - harmony[
#                         i]) * random.random() * self._obj_fun.get_mpap()
#
#
# class UpdateMemoryStrategy(ABC):
#     @abstractmethod
#     def update_harmony_memory(self, considered_harmony, considered_fitness):
#         pass
#
# class UpdateMemoryVanilla(UpdateMemoryStrategy):
#     def _update_harmony_memory(self, considered_harmony, considered_fitness):
#         """
#             Update the harmony memory if necessary with the given harmony. If the given harmony is better than the worst
#             harmony in memory, replace it. This function doesn't allow duplicate harmonies in memory.
#         """
#         if (considered_harmony, considered_fitness) not in self._harmony_memory and self._obj_fun.check_constraints(
#                 considered_harmony):
#             worst_index = None
#             worst_fitness = float('+inf') if self._obj_fun.maximize() else float('-inf')
#             for i, (harmony, fitness) in enumerate(self._harmony_memory):
#                 if (self._obj_fun.maximize() and fitness < worst_fitness) or (
#                         not self._obj_fun.maximize() and fitness > worst_fitness):
#                     worst_index = i
#                     worst_fitness = fitness
#             if (self._obj_fun.maximize() and considered_fitness > worst_fitness) or (
#                     not self._obj_fun.maximize() and considered_fitness < worst_fitness):
#                 self._harmony_memory[worst_index] = (considered_harmony, considered_fitness)
