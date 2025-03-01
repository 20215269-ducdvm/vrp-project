import copy
from abc import ABC, abstractmethod


class RunStrategy(ABC):
    @abstractmethod
    def run(self, initial_harmonies=None):
        pass


class InitializationStrategy(ABC):
    @abstractmethod
    def initialize(self, initial_harmonies=None):
        pass


class VanillaInitialization(InitializationStrategy):
    def initialize(self, initial_harmonies=None):
        """
            Initialize harmony_memory, the matrix (list of lists) containing the various harmonies (solution vectors). Note
            that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
            merely stores previous harmonies.

            If harmonies are provided, then use them instead of randomly initializing them.

            Populate harmony_history with initial harmony memory.
        """
        if initial_harmonies is not None:
            # verify that the initial harmonies are provided correctly

            if len(initial_harmonies) != self._obj_fun.get_hms():
                raise ValueError('Number of initial harmonies does not equal to the harmony memory size.')

            num_parameters = self._obj_fun.get_num_parameters()
            for i in range(len(initial_harmonies)):
                num_parameters_initial_harmonies = len(initial_harmonies[i])
                if num_parameters_initial_harmonies != num_parameters:
                    raise ValueError('Number of parameters in initial harmonies does not match that defined.')
        else:
            initial_harmonies = list()
            for i in range(0, self._obj_fun.get_hms()):
                harmony = list()
                for j in range(0, self._obj_fun.get_num_parameters()):
                    self._random_selection(harmony, j)
                initial_harmonies.append(harmony)

        for i in range(0, self._obj_fun.get_hms()):
            fitness = self._obj_fun.get_fitness(initial_harmonies[i])
            self._harmony_memory.append((initial_harmonies[i], fitness))

        harmony_list = {'gen': 0, 'harmonies': copy.deepcopy(self._harmony_memory)}
        self._harmony_history.append(harmony_list)


class PitchAdjustmentStrategy(ABC):
    @abstractmethod
    def pitch_adjustment(self, harmony, i):
        pass


class UpdateMemoryStrategy(ABC):
    @abstractmethod
    def update_harmony_memory(self, considered_harmony, considered_fitness):
        pass
