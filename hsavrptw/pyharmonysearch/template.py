import random
from abc import abstractmethod, ABC


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
