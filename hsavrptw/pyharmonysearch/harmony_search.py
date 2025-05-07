"""
    Copyright (c) 2013, Triad National Security, LLC
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
      disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the type of Triad National Security, LLC nor the names of its contributors may be used to endorse or
      promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
    THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import copy
import random
from collections import namedtuple
from datetime import datetime
from multiprocessing import Pool, Event
import logging
logger = logging.getLogger(__name__)
# Note: We use a global multiprocessing.Event to deal with a KeyboardInterrupt. This idea comes from
# http://stackoverflow.com/questions/14579474/multiprocessing-pool-spawning-new-childern-after-terminate-on-linux-python2-7.
# This is not necessary when running under Python 3, but to keep 2.7 compatability, I'm leaving it in.
terminating = Event()

# HarmonySearchResults is a struct-like object that we'll use to attach the results of the search.
# namedtuples are lightweight and trivial to extend should more results be desired in the future. Right now, we're just
# keeping track of the total elapsed clock time, the best solution found, the fitness for that solution, and the solution memory,
# which allows you to see the top harmonies.
HarmonySearchResults = namedtuple('HarmonySearchResults',
                                  ['elapsed_time', 'best_harmony', 'best_fitness', 'harmony_memories',
                                   'harmony_histories'])


from .template import HarmonySearchTemplate

def harmony_search(objective_function, num_processes, num_iterations, algorithm: HarmonySearchTemplate,
                   initial_harmonies=None):
    """
        Here, we use multiprocessing.Pool to do multiple solution searches simultaneously. Since HS is stochastic (unless random_seed is set),
        multiple runs can find different results. We run the specified number of iterations on the specified number of processes and return
        an instance of HarmonySearchResults.
    """
    pool = Pool(num_processes)
    try:
        start = datetime.now()
        pool_results = [pool.apply_async(worker, args=(algorithm, initial_harmonies,)) for i in
                        range(num_iterations)]
        pool.close()  # no more tasks will be submitted to the pool
        pool.join()  # wait for all tasks to finish before moving on
        end = datetime.now()
        elapsed_time = end - start

        # find best solution from all iterations
        best_harmony = None
        best_fitness = float('-inf') if objective_function.maximize() else float('+inf')
        harmony_memories = list()
        harmony_histories = list()
        for result in pool_results:
            harmony, fitness, harmony_memory, harmony_history = result.get()  # multiprocessing.pool.AsyncResult is returned for each process, so we need to call get() to pull out the value
            if (objective_function.maximize() and fitness > best_fitness) or (
                    not objective_function.maximize() and fitness < best_fitness):
                best_harmony = harmony
                best_fitness = fitness
            harmony_memories.append(harmony_memory)
            harmony_histories.append(harmony_history)

        return HarmonySearchResults(elapsed_time=elapsed_time, best_harmony=best_harmony, best_fitness=best_fitness, \
                                    harmony_memories=harmony_memories, harmony_histories=harmony_histories)
    except KeyboardInterrupt:
        pool.terminate()
        raise


def harmony_search_serial(objective_function, num_iterations, algorithm: HarmonySearchTemplate, initial_harmonies=None):
    """
        Same as ``harmony_search`` but without multiprocessing. This could be useful when there's already multiprocessing in, e.g.,
        ``get_fitness`` method in ``objective_function``, since multiprocessing cannot be used within multiprocessing.
    """
    start = datetime.now()
    results = [worker(algorithm, initial_harmonies) for i in range(num_iterations)]
    end = datetime.now()
    elapsed_time = end - start

    # find best solution from all iterations
    best_harmony = None
    best_fitness = float('-inf') if objective_function.maximize() else float('+inf')
    harmony_memories = list()
    harmony_histories = list()
    for result in results:
        harmony, fitness, harmony_memory, harmony_history = result
        if (objective_function.maximize() and fitness > best_fitness) or (
                not objective_function.maximize() and fitness < best_fitness):
            best_harmony = harmony
            best_fitness = fitness
        harmony_memories.append(harmony_memory)
        harmony_histories.append(harmony_history)

    return HarmonySearchResults(elapsed_time=elapsed_time, best_harmony=best_harmony, best_fitness=best_fitness, \
                                harmony_memories=harmony_memories, harmony_histories=harmony_histories)


def worker(algorithm, initial_harmonies=None):
    """
        This is just a dummy function to make multiprocessing work with a class. It also checks/sets the global multiprocessing.Event to prevent
        new processes from starting work on a KeyboardInterrupt.
    """
    try:
        if not terminating.is_set():
            return algorithm.run(initial_harmonies=initial_harmonies)
    except KeyboardInterrupt:
        terminating.set()  # set the Event to true to prevent the other processes from doing any work
        raise


def insert_harmony_to_memory(harmony_memory, new_harmony, fitness):
    """
    Insert a new solution into the solution memory in the appropriate position.

    :param harmony_memory: List of tuples (solution, fitness)
    :param new_harmony: The new solution to insert
    :param fitness: The fitness value of the new solution

    :return:
        Updated solution memory with the new solution inserted
    """
    if not harmony_memory:
        return [(new_harmony, fitness)]

    # create the new entry
    new_entry = (new_harmony, fitness)

    # copy the memory to avoid modifying the original
    updated_memory = harmony_memory.copy()

    # split the memory into valid (fitness > 0) and invalid (fitness < 0) solutions
    valid_solutions = [entry for entry in updated_memory if entry[1] > 0]
    invalid_solutions = [entry for entry in updated_memory if entry[1] < 0]

    # insert the new entry based on its fitness value
    if fitness > 0:  # valid curr_solution
        # insert into valid solutions in ascending order
        inserted = False
        for i, (_, existing_fitness) in enumerate(valid_solutions):
            if fitness < existing_fitness:  # found the position
                valid_solutions.insert(i, new_entry)
                inserted = True
                break
        if not inserted:  # if it's larger than all existing values
            valid_solutions.append(new_entry)

    else:  # invalid curr_solution (fitness < 0)
        # insert into invalid solutions in descending order
        inserted = False
        for i, (_, existing_fitness) in enumerate(invalid_solutions):
            if fitness > existing_fitness:  # found the position
                invalid_solutions.insert(i, new_entry)
                inserted = True
                break
        if not inserted:  # if it's smaller than all existing values
            invalid_solutions.append(new_entry)

    # combine valid and invalid solutions
    updated_memory = valid_solutions + invalid_solutions

    return updated_memory


class HarmonySearchVRP(HarmonySearchTemplate):
    """
        This class implements the solution search (HS) global optimization algorithm. In general, what you'll do is this:

        1. Implement an objective function that inherits from ObjectiveFunctionInterface.
        2. Initialize HarmonySearch with this objective function (e.g., hs = HarmonySearch(objective_function)).
        3. Run HarmonySearch (e.g., results = hs.run()).
    """
    def run(self, initial_harmonies=None):
        """
            This is the main HS loop. It initializes the solution memory and then continually generates new harmonies
            until the stopping criterion (max_imp iterations) is reached.
        """
        # set optional random seed
        if self._obj_fun.use_random_seed():
            random.seed(self._obj_fun.get_random_seed())

        # harmony_memory stores the best hms harmonies
        self._harmony_memory = list()

        # harmony_history stores all hms harmonies every nth improvisations (i.e., one 'generation')
        self._harmony_history = list()

        # fill harmony_memory using random parameter values by default, but with initial_harmonies if provided
        self.initialize(initial_harmonies)

        # create max_imp improvisations
        generation = 0
        num_imp = 0
        num_discrete_values: int = 0
        while num_imp < self._obj_fun.get_max_imp():
            # generate new solution
            harmony = list()

            for i in range(0, self._obj_fun.get_num_parameters()):
                num_discrete_values = self._obj_fun.get_num_discrete_values(i) - 1
                if random.random() < self._obj_fun.get_hmcr():
                    self.memory_consideration(harmony, i)
                    if random.random() < self._obj_fun.get_par():
                        self.pitch_adjustment(harmony, i)
                else:
                    self.random_selection(harmony, i)


            # after generating a new solution, check the validity of the solution format
            # and fix it if necessary
            # solution = remove_duplicates(solution, upper_bound=num_discrete_values)

            fitness = self._obj_fun.get_fitness(harmony)
            self.update_harmony_memory(harmony, fitness)
            num_imp += 1

            # save harmonies every nth improvisations (i.e., one 'generation')
            if num_imp % self._obj_fun.get_hms() == 0:
                generation += 1
                harmony_list = {'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)}
                self._harmony_history.append(harmony_list)

        # return best solution
        best_harmony = None
        best_fitness = float('-inf') if self._obj_fun.maximize() else float('+inf')
        for harmony, fitness in self._harmony_memory:
            if (self._obj_fun.maximize() and fitness > best_fitness) or (
                    not self._obj_fun.maximize() and fitness < best_fitness):
                best_harmony = harmony
                best_fitness = fitness
                logger.info(f"Best solution: {best_harmony}, Fitness: {best_fitness}, Number of routes: {len([x for x in best_harmony if x == 0])}")

        return best_harmony, best_fitness, self._harmony_memory, self._harmony_history

    def initialize(self, initial_harmonies=None):
        """
            Initialize harmony_memory, the matrix (list of lists) containing the various harmonies (curr_solution vectors). Note
            that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
            merely stores previous harmonies.

            If harmonies are provided, then use them instead of randomly initializing them.

            If a generated solution is infeasible, we calculate the degree of violation and use it as the fitness value.

            The solution memory is divided into two parts: feasible and infeasible harmonies. The feasible harmonies are
            stored in the first part of the memory, while the infeasible harmonies are stored in the second part. The
            infeasible harmonies are sorted in ascending order of the degree of violation. The worst infeasible solution
            is the one with the highest degree of violation.

            Populate harmony_history with initial solution memory.
        """
        if initial_harmonies is not None:
            # verify that the initial harmonies are provided correctly

            if len(initial_harmonies) != self._obj_fun.get_hms():
                raise ValueError('Number of initial harmonies does not equal to the solution memory size.')

            num_parameters = self._obj_fun.get_num_parameters()
            for i in range(len(initial_harmonies)):
                num_parameters_initial_harmonies = len(initial_harmonies[i])
                if num_parameters_initial_harmonies != num_parameters:
                    raise ValueError('Number of parameters in initial harmonies does not match that defined.')
        else:
            # MAIN PART
            initial_harmonies = list()
            for i in range(0, self._obj_fun.get_hms()):
                while True:
                    harmony = list()
                    for j in range(0, self._obj_fun.get_num_parameters()):
                        self.random_selection(harmony, j)
                    fitness = self._obj_fun.get_fitness(harmony)
                    if fitness > 0:
                        break
                initial_harmonies = insert_harmony_to_memory(initial_harmonies, harmony, fitness)


        self._harmony_memory = initial_harmonies

        harmony_list = {'gen': 0, 'harmonies': copy.deepcopy(self._harmony_memory)}
        self._harmony_history.append(harmony_list)

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

    def update_harmony_memory(self, considered_harmony, considered_fitness):
        """
        Update the solution memory if necessary with the given solution.
        The decision process depends on the feasibility of the considered solution.
        :param considered_harmony: The new solution to consider
        :param considered_fitness: The fitness value of the new solution
        """
        # check if the solution is already in memory
        if (considered_harmony, considered_fitness) in self._harmony_memory or considered_fitness == float('-inf'):
            return  # no duplicates allowed

        # split memory into valid (fitness > 0) and invalid (fitness < 0) solutions
        valid_solutions = [(h, f) for h, f in self._harmony_memory if f > 0]
        invalid_solutions = [(h, f) for h, f in self._harmony_memory if f <= 0]

        # case 1: considered solution is feasible (fitness > 0)
        if considered_fitness > 0:
            # find the appropriate position to insert in valid solutions (ascending order)
            insertion_index = len(valid_solutions)
            for i, (_, fitness) in enumerate(valid_solutions):
                if considered_fitness < fitness:  # found position (ascending order)
                    insertion_index = i
                    break

            # insert into valid solutions
            valid_solutions.insert(insertion_index, (considered_harmony, considered_fitness))

            # remove worst infeasible solution if exists
            if invalid_solutions:
                # for infeasible solutions, the worst is the one with smallest fitness (most negative)
                invalid_solutions = invalid_solutions[:-1]  # remove the last element (worst one)

        # case 2: considered solution is infeasible (fitness <= 0)
        else:
            # check if the worst solution in memory is feasible
            if not self._harmony_memory:  # empty memory case
                invalid_solutions.append((considered_harmony, considered_fitness))
            elif len(valid_solutions) == len(self._harmony_memory):  # all harmonies are feasible
                # don't insert the considered solution
                pass
            else:  # there are some infeasible harmonies
                # find the appropriate position to insert in invalid solutions (descending order)
                insertion_index = len(invalid_solutions)
                for i, (_, fitness) in enumerate(invalid_solutions):
                    if considered_fitness > fitness:  # found position (descending order)
                        insertion_index = i
                        break

                # insert into invalid solutions
                if insertion_index < len(invalid_solutions):
                    invalid_solutions.insert(insertion_index, (considered_harmony, considered_fitness))
                    # remove the worst infeasible solution (the last one after sorting)
                    invalid_solutions = invalid_solutions[:-1]

        # update solution memory with the modified valid and invalid solutions
        self._harmony_memory = valid_solutions + invalid_solutions

