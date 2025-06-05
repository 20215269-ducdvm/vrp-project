import logging
import random
from abc import abstractmethod, ABC

logger = logging.getLogger(__name__)


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
    def initialize(self, initial_harmonies=None):
        pass

    def random_selection(self, harmony, i):
        """
            Choose a random note from the possible values.
        """
        if self._obj_fun.is_discrete(i):
            harmony.append(self._obj_fun.get_value(i, random.randint(0, self._obj_fun.get_num_discrete_values(i) - 1)))
        else:
            harmony.append(random.uniform(self._obj_fun.get_lower_bound(i), self._obj_fun.get_upper_bound(i)))

    def memory_consideration(self, harmony, i):
        """
            Randomly choose a note previously played.
        """
        memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
        harmony.append(self._harmony_memory[memory_index][0][i])

    @abstractmethod
    def pitch_adjustment(self, harmony, i):
        """
            If variable, randomly adjust the pitch up or down by some amount. This is the only place in the algorithm where there
            is an explicit difference between continuous and discrete variables.

            The probability of adjusting the pitch either up or down is fixed at 0.5. The maximum pitch adjustment proportion (mpap)
            and maximum pitch adjustment index (mpai) determine the maximum amount the pitch may change for continuous and discrete
            variables, respectively.

            For example, suppose that it is decided via coin flip that the pitch will be adjusted down. Also suppose that mpap is set to 0.25.
            This means that the maximum value the pitch can be dropped will be 25% of the difference between the lower bound and the current
            pitch. mpai functions similarly, only it relies on indices of the possible values instead.
        """
        if self._obj_fun.is_variable(i):
            if self._obj_fun.is_discrete(i):
                current_index = self._obj_fun.get_index(i, harmony[i])
                # discrete variable
                if random.random() < 0.5:
                    # adjust pitch down
                    harmony[i] = self._obj_fun.get_value(i, current_index - random.randint(0,
                                                                                           min(self._obj_fun.get_mpai(),
                                                                                               current_index)))
                else:
                    # adjust pitch up
                    harmony[i] = self._obj_fun.get_value(i, current_index + random.randint(0,
                                                                                           min(self._obj_fun.get_mpai(),
                                                                                               self._obj_fun.get_num_discrete_values(
                                                                                                   i) - current_index - 1)))
            else:
                # continuous variable
                if random.random() < 0.5:
                    # adjust pitch down
                    harmony[i] -= (harmony[i] - self._obj_fun.get_lower_bound(
                        i)) * random.random() * self._obj_fun.get_mpap()
                else:
                    # adjust pitch up
                    harmony[i] += (self._obj_fun.get_upper_bound(i) - harmony[
                        i]) * random.random() * self._obj_fun.get_mpap()

    @abstractmethod
    def update_harmony_memory(self, considered_harmony, considered_fitness):
        """
                    Update the solution memory if necessary with the given solution. If the given solution is better than the worst
                    solution in memory, replace it. This function doesn't allow duplicate harmonies in memory.
                """
        if (considered_harmony, considered_fitness) not in self._harmony_memory:
            worst_index = None
            worst_fitness = float('+inf') if self._obj_fun.maximize() else float('-inf')
            for i, (harmony, fitness) in enumerate(self._harmony_memory):
                if (self._obj_fun.maximize() and fitness < worst_fitness) or (
                        not self._obj_fun.maximize() and fitness > worst_fitness):
                    worst_index = i
                    worst_fitness = fitness
            if (self._obj_fun.maximize() and considered_fitness > worst_fitness) or (
                    not self._obj_fun.maximize() and considered_fitness < worst_fitness):
                self._harmony_memory[worst_index] = (considered_harmony, considered_fitness)
