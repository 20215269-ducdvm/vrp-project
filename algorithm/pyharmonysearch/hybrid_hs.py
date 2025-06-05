import copy
import logging
import random
import time

from datastructure import VRPProblem, VRPSolution
from helpers.helpers import remove_duplicates, compute_euclidean_distance
from pyharmonysearch.harmony_search import HarmonySearchVRP, insert_harmony_to_memory

logger = logging.getLogger(__name__)

def decode_harmony_to_routes(harmony):
    """
    Decode a solution into a list of routes.
    Args:
        harmony: the solution vector

    Returns:
        A list of routes, including depot
    """
    routes = []

    route = [0]
    for i in range(len(harmony)):
        route.append(harmony[i])
        if harmony[i] == 0:
            if len(route) > 2:  # Not an empty route (has more than just depot-depot)
                routes.append(route)
            route = [0]  # Start a new route with depot

    # Check if there's a final route that didn't end with depot
    if len(route) > 1:
        route.append(0)  # Close the route with depot
        routes.append(route)

    return routes


def encode_routes_to_harmony(routes: list[list[int]]) -> list[int]:
    """
    Convert a list of routes back to a solution (encoded solution).

    Args:
        routes: list of routes, where each route is a list of customer indices

    Returns:
        Encoded solution (solution)
    """
    harmony = []

    for i, route in enumerate(routes):
        # Add all customers except depot at the end
        for node in route[1:-1]:
            harmony.append(node)

        # Add vehicle separator if not the last route
        if i < len(routes) - 1:
            harmony.append(0)

    return harmony


class HybridHeuristicHSA(HarmonySearchVRP):
    """
    Hybrid Heuristic Harmony Search Algorithm for solving the Vehicle Routing Problem with Time Windows (VRPTW).
    It generates solutions using Harmony Search principles and applies a local search method (KGLS) to improve them.
    """
    def __init__(self, obj_fun):
        """
        Initialize the Hybrid Heuristic Harmony Search Algorithm.

        Args:
            obj_fun: Objective function for the VRPTW
        """
        super().__init__(obj_fun)
        self.max_zeros_allowed = obj_fun.problem_instance.number_vehicles_required - 1

        # assert kgls is not None, "KGLS instance must be provided for local search."
        # self.kgls = kgls  # KGLS instance for local search

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
        while num_imp < self._obj_fun.get_max_imp():
            # generate new solution
            harmony = list()

            for i in range(self._obj_fun.get_num_parameters()):
                if random.random() < self._obj_fun.get_hmcr():
                    self.memory_consideration(harmony, i)
                    if random.random() < self._obj_fun.get_par():
                        self.pitch_adjustment(harmony, i)
                else:
                    self.random_selection(harmony, i)

            harmony = self._improve_harmony(harmony)

            fitness = self._obj_fun.get_fitness(harmony)

            self.update_harmony_memory(harmony, fitness)

            num_imp += 1

            # save harmonies every nth improvisations (i.e., one 'generation')
            if num_imp % self._obj_fun.get_hms() == 0:
                generation += 1
                harmony_list = {'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)}

                self._harmony_history.append(harmony_list)

        # return best solution
        best_harmony, best_fitness, best_violation = self._return_best_harmony()

        logger.info(f"Cost: {best_fitness}, violation: {best_violation}")
        return best_harmony, best_fitness, self._harmony_memory, self._harmony_history

    def _apply_local_search(self, solution: VRPSolution) -> 'VRPSolution':
        """
        Apply KGLS to improve a solution.

        Args:
            solution: The VRPSolution to improve

        Returns:
            Tuple of (improved solution, solution cost)
        """
        start_time = time.time()

        # Apply local search improvement using KGLS instance
        self.kgls.run(visualize_progress=False, start_solution=solution)
        improved_solution = self.kgls.best_solution

        # TODO: Solution stats
        return improved_solution

    def _improve_harmony(self, harmony):
        """
        Improve the harmony using a local search method.

        This method can be overridden to implement specific local search strategies.
        """
        harmony = remove_duplicates(harmony,
                                    upper_bound=self._obj_fun.get_num_discrete_values(0) - 1,
                                    max_zeros_allowed=self._obj_fun.problem_instance.number_vehicles_required - 1)

        # TODO Apply local search to the generated solution
        # to either restore feasibility for infeasible solutions, or minimize total costs for feasible ones

        # solution = vector_to_vrp_solution(harmony, self._obj_fun.problem_instance)
        #
        # improved_solution = self._apply_local_search(solution)
        #
        # harmony = vrp_solution_to_vector(improved_solution)

        return harmony

    def _return_best_harmony(self):
        """
        Return the best harmony from the memory.

        This is a helper function to retrieve the best harmony after the search process.

        Returns:
            The best harmony and its fitness value
        """
        best_harmony = None
        best_fitness = float('+inf')
        best_violation = float('+inf')
        for harmony, fitness in self._harmony_memory:
            # Check if the solution is feasible (violation_degree == 0)
            if fitness[1] <= 0:
                # For feasible solutions, compare the distance (fitness[0])
                if fitness[0] < best_fitness:
                    best_harmony = harmony
                    best_fitness = fitness[0]

        # If no feasible solution was found, select the solution with the smallest violation
        if best_harmony is None:
            for harmony, fitness in self._harmony_memory:
                if fitness[1] < best_violation:
                    best_harmony = harmony
                    best_fitness = fitness[0]
                    best_violation = fitness[1]

        return best_harmony, best_fitness, best_violation

    def initialize(self, initial_harmonies=None):
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
            for i in range(self._obj_fun.get_hms()):
                while True:
                    harmony = list()
                    for j in range(self._obj_fun.get_num_parameters()):
                        self.random_selection(harmony, j)
                    harmony = remove_duplicates(harmony, upper_bound=self._obj_fun.get_num_discrete_values(0) - 1,
                                                max_zeros_allowed=self.max_zeros_allowed)
                    fitness = self._obj_fun.get_fitness(harmony)
                    if fitness[0] > float('-inf'):
                        break
                initial_harmonies = insert_harmony_to_memory(initial_harmonies, harmony, fitness)

        self._harmony_memory = initial_harmonies

        harmony_list = {'gen': 0, 'harmonies': copy.deepcopy(self._harmony_memory)}
        self._harmony_history.append(harmony_list)

    def random_selection(self, harmony, i):
        """
        Randomly choose a value for the current component.

        Additional rule: don't generate too many 0's. If the 0's count already equals
        or exceeds the number of vehicles required minus 1, generate a non-zero note.

        Args:
            harmony: Current (partial) solution
            i: Current component index
        """
        # Get the maximum possible value for this position
        max_value = self._obj_fun.get_num_discrete_values(i) - 1

        # Count existing zeros in the current harmony
        zeros_count = harmony.count(0)

        # Check if we already have enough zeros
        if zeros_count >= self.max_zeros_allowed:
            # If we already have enough zeros, generate a non-zero value
            if max_value > 0:  # Make sure there's at least one customer
                selected_note = random.randint(1, max_value)
            else:
                # If there are no valid customers, use 0 as a last resort
                selected_note = 0
        else:
            # We can select any value including 0
            selected_note = random.randint(0, max_value)

        harmony.append(selected_note)

    def memory_consideration(self, harmony, i):
        """
        Randomly choose a note previously played.

        Additional rule: don't generate too much 0's. If the 0's count exceed the number required,
        generate another note different from 0.
        Args:
            harmony: Current (partial) solution
            i: Current component index
        """
        memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
        selected_note = self._harmony_memory[memory_index][0][i]
        zeros_count = harmony.count(0)

        # Check if we already have enough zeros
        if selected_note == 0 and zeros_count >= self.max_zeros_allowed:
            # Try up to 3 times to find a non-zero note from memory
            for _ in range(3):
                alt_memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
                if i < len(self._harmony_memory[alt_memory_index][0]):
                    alternative_note = self._harmony_memory[alt_memory_index][0][i]
                    if alternative_note != 0:
                        selected_note = alternative_note
                        break

            # If we still have a zero, generate a random non-zero note
            if selected_note == 0:
                max_customer = self._obj_fun.get_num_discrete_values(i) - 1
                if max_customer > 0:  # Make sure there's at least one customer
                    selected_note = random.randint(1, max_customer)

        harmony.append(selected_note)

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
            current_index = self._obj_fun.get_index(i, harmony[i])
            zeros_count = harmony.count(0)

            # For discrete variables
            if random.random() < 0.5:
                # adjust pitch down
                new_index = current_index - random.randint(0, min(self._obj_fun.get_mpai(), current_index))

                # Make sure we don't create a zero if we already have enough
                if self._obj_fun.get_value(i, new_index) == 0 and zeros_count >= self.max_zeros_allowed:
                    # Find the closest non-zero value
                    for idx in range(new_index + 1, current_index + 1):
                        val = self._obj_fun.get_value(i, idx)
                        if val != 0:
                            new_index = idx
                            break

                harmony[i] = self._obj_fun.get_value(i, new_index)
            else:
                # adjust pitch up
                harmony[i] = self._obj_fun.get_value(i, current_index + random.randint(0,
                                                                                       min(self._obj_fun.get_mpai(),
                                                                                           self._obj_fun.get_num_discrete_values(
                                                                                               i) - current_index - 1)))


def vrp_solution_to_vector(solution: VRPSolution) -> list[int]:
    """
    Convert a VRP solution back to a solution vector.

    Args:
        solution: The VRPSolution to convert

    Returns:
        A vector representation of the solution
    """
    vector = []

    for route in solution.routes:
        if route.size > 0:
            for node in route.customers:
                vector.append(node.node_id)
            vector.append(0)  # Add depot to mark end of route

    return vector


def vector_to_vrp_solution(vector, problem: VRPProblem) -> VRPSolution:
    """
    Convert a solution vector to a VRP solution.

    Args:
        vector: The solution vector to convert
        problem: The VRPProblem instance

    Returns:
        A VRPSolution instance
    """

    solution = VRPSolution(problem)

    # Parse the vector into routes
    current_route = []
    depot = 0  # Assuming depot is represented by 0

    for node_id in vector:
        if node_id == depot:
            if current_route:
                # Add completed route
                solution.add_route([problem.nodes[i] for i in current_route])
                current_route = []
        else:
            current_route.append(node_id)

    # Add any remaining nodes
    if current_route:
        solution.add_route([problem.nodes[i] for i in current_route])

    return solution
