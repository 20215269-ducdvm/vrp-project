import copy
import random
from abc import abstractmethod, ABC

# from pyharmonysearch.strategies import InitializationVanilla, PitchAdjustmentVanilla, UpdateMemoryVanilla

# TODO: Make code pretty later
# class HarmonySearchBuilder:
#     def __init__(self, objective_function):
#         self._objective_function = objective_function
#         self._initialization_strategy = InitializationVanilla()
#         self._pitch_adjustment_strategy = PitchAdjustmentVanilla()
#         self._update_memory_strategy = UpdateMemoryVanilla()
#
#     def set_initialization_strategy(self, initialization_strategy):
#         self._initialization_strategy = initialization_strategy
#         return self
#
#     def set_pitch_adjustment_strategy(self, pitch_adjustment_strategy):
#         self._pitch_adjustment_strategy = pitch_adjustment_strategy
#         return self
#
#     def set_update_memory_strategy(self, update_memory_strategy):
#         self._update_memory_strategy = update_memory_strategy
#         return self
#
#     def build(self):
#         return HarmonySearchTemplate(
#             self._objective_function,
#             self._initialization_strategy,
#             self._pitch_adjustment_strategy,
#             self._update_memory_strategy
#         )
#
# class HarmonySearch:
#     def __init__(self, objective_function, initialization_strategy, pitch_adjustment_strategy, update_memory_strategy):
#         self._harmony_memory = None
#         self._obj_fun = objective_function
#         self._initialization_strategy = initialization_strategy
#         self._pitch_adjustment_strategy = pitch_adjustment_strategy
#         self._update_memory_strategy = update_memory_strategy
#
#     def run(self, initial_harmonies=None):
#         """
#             This is the main HS loop. It initializes the harmony memory and then continually generates new harmonies
#             until the stopping criterion (max_imp iterations) is reached.
#         """
#         # set optional random seed
#         if self._obj_fun.use_random_seed():
#             random.seed(self._obj_fun.get_random_seed())
#
#         # harmony_memory stores the best hms harmonies
#         self._harmony_memory = list()
#
#         # harmony_history stores all hms harmonies every nth improvisations (i.e., one 'generation')
#         self._harmony_history = list()
#
#         # fill harmony_memory using random parameter values by default, but with initial_harmonies if provided
#         self._initialize(initial_harmonies)
#
#         # create max_imp improvisations
#         generation = 0
#         num_imp = 0
#         while (num_imp < self._obj_fun.get_max_imp()):
#             # generate new harmony
#             harmony = list()
#             for i in range(0, self._obj_fun.get_num_parameters()):
#                 if random.random() < self._obj_fun.get_hmcr():
#                     self._memory_consideration(harmony, i)
#                     if random.random() < self._obj_fun.get_par():
#                         self._pitch_adjustment(harmony, i)
#                 else:
#                     self._random_selection(harmony, i)
#             fitness = self._obj_fun.get_fitness(harmony)
#             self._update_harmony_memory(harmony, fitness)
#             num_imp += 1
#
#             # save harmonies every nth improvisations (i.e., one 'generation')
#             if num_imp % self._obj_fun.get_hms() == 0:
#                 generation += 1
#                 harmony_list = {'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)}
#                 self._harmony_history.append(harmony_list)
#
#         # return best harmony
#         best_harmony = None
#         best_fitness = float('-inf') if self._obj_fun.maximize() else float('+inf')
#         for harmony, fitness in self._harmony_memory:
#             if (self._obj_fun.maximize() and fitness > best_fitness) or (
#                     not self._obj_fun.maximize() and fitness < best_fitness):
#                 best_harmony = harmony
#                 best_fitness = fitness
#         return best_harmony, best_fitness, self._harmony_memory, self._harmony_history
#
#     def _initialize(self, initial_harmonies=None):
#         self._initialization_strategy.initialize(initial_harmonies)
#
#     def _random_selection(self, harmony, i):
#         if self._obj_fun.is_discrete(i):
#             harmony.append(self._obj_fun.get_value(i, random.randint(0, self._obj_fun.get_num_discrete_values(i) - 1)))
#         else:
#             harmony.append(random.uniform(self._obj_fun.get_lower_bound(i), self._obj_fun.get_upper_bound(i)))
#
#     def _memory_consideration(self, harmony, i):
#         memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
#         harmony.append(self._harmony_memory[memory_index][0][i])
#
#     def _pitch_adjustment(self, harmony, i):
#         self._pitch_adjustment_strategy.pitch_adjustment(harmony, i)
#
#     def _update_harmony_memory(self, considered_harmony, considered_fitness):
#         self._update_memory_strategy.update_harmony_memory(considered_harmony, considered_fitness)

class HarmonySearchTemplate(ABC):
    def __init__(self, objective_function):
        """
            Initialize HS with the specified objective function. Note that this objective function must implement ObjectiveFunctionInterface.
        """
        self._harmony_memory = None
        self._obj_fun = objective_function

    @abstractmethod
    def run(self, initial_harmonies=None):
        pass

    @abstractmethod
    def _initialize(self, initial_harmonies=None):
        pass

    def _random_selection(self, harmony, i):
        """
            Choose a random note from the possible values.
        """
        if self._obj_fun.is_discrete(i):
            harmony.append(self._obj_fun.get_value(i, random.randint(0, self._obj_fun.get_num_discrete_values(i) - 1)))
        else:
            harmony.append(random.uniform(self._obj_fun.get_lower_bound(i), self._obj_fun.get_upper_bound(i)))

    def _memory_consideration(self, harmony, i):
        """
            Randomly choose a note previously played.
        """
        memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
        harmony.append(self._harmony_memory[memory_index][0][i])

    @abstractmethod
    def _pitch_adjustment(self, harmony, i):
        pass

    @abstractmethod
    def _update_harmony_memory(self, considered_harmony, considered_fitness):
        pass
