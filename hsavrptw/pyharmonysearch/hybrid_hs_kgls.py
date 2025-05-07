import logging
import random
import time
from collections import defaultdict

from datastructure import VRPSolution
from helpers.helpers import remove_duplicates
from datastructure.vrp_objective_function import vrp_solution_to_vector, vector_to_vrp_solution
from heuristics.guided_local_search.kgls import KGLS
from heuristics.guided_local_search.kgls.kgls import DEFAULT_PARAMETERS
from heuristics.guided_local_search.kgls.solution_construction import clark_wright_route_reduction
from pyharmonysearch.harmony_search import HarmonySearchVRP

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridHSKGLS(HarmonySearchVRP):
    """
    Hybrid Harmony Search algorithm with KGLS improvement.

    This class extends the HarmonySearchVRP algorithm by incorporating KGLS (Kernel-based
    Guided Local Search) to improve solutions during the solution search process.
    """

    def __init__(self, objective_function, kgls=None, to_file=None):
        """
        Initialize the hybrid algorithm.

        Args:
            objective_function: The objective function to optimize
            kgls: KGLS instance to use for solution improvement
        """
        super().__init__(objective_function)
        self._problem_instance = objective_function._problem_instance
        self._write_to_file = to_file

        # Store KGLS instance
        self.kgls = kgls
        if kgls is None:
            # Create a new KGLS instance if none was provided
            self.kgls = KGLS(path_to_instance_file="", **DEFAULT_PARAMETERS)
            self.kgls._vrp_instance = self._problem_instance

        # Use the cost evaluator from KGLS
        self._cost_evaluator = self.kgls.cost_evaluator
        self._solution_stats = defaultdict(float)

    def _apply_kgls(self, solution: VRPSolution) -> tuple[list, float]:
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

        # Calculate solution cost
        improved_harmony = vrp_solution_to_vector(improved_solution)
        cost = self._obj_fun.get_fitness(improved_harmony)

        # Update stats
        self._solution_stats["time_kgls_improvement"] += time.time() - start_time

        return improved_harmony, cost

    def initialize(self, initial_harmonies=None):
        """
        Initialize solution memory with high-quality solutions improved by KGLS.
        """
        logger.info("Initializing solution memory with KGLS-improved solutions")

        self._harmony_memory = []

        # If user provided initial harmonies, use them
        if initial_harmonies is not None:
            # Verify initial harmonies
            if len(initial_harmonies) != self._obj_fun.get_hms():
                raise ValueError('Number of initial harmonies does not equal the solution memory size.')

            self._harmony_memory = initial_harmonies

            num_parameters = self._obj_fun.get_num_parameters()
            for i in range(len(initial_harmonies)):
                num_parameters_initial_harmonies = len(initial_harmonies[i][0])
                if num_parameters_initial_harmonies != num_parameters:
                    raise ValueError('Number of parameters in initial harmonies does not match that defined.')

            # Convert to VRP solutions, improve with KGLS, and add to memory
            # for solution in initial_harmonies:
            #     vrp_solution = vector_to_vrp_solution(solution, self._problem_instance)
            #     improved_solution, fitness = self._apply_kgls(vrp_solution)
            #
            #     self._harmony_memory.append((improved_solution, fitness))

        else:
            # Generate new solutions
            # First solution using Clark-Wright
            logger.info("Generating first solution using Clark-Wright")
            cw_solution = clark_wright_route_reduction(vrp_instance=self._problem_instance,
                                                       cost_evaluator=self._cost_evaluator)
            improved_vector, fitness = self._apply_kgls(cw_solution)

            self._harmony_memory.append((improved_vector, fitness))

            # Generate remaining solutions randomly and improve with KGLS
            for i in range(1, self._obj_fun.get_hms()):
                logger.info(f"Generating random solution {i}/{self._obj_fun.get_hms() - 1}")
                harmony = []
                for j in range(self._obj_fun.get_num_parameters()):
                    self.random_selection(harmony, j)

                vrp_solution = vector_to_vrp_solution(harmony)
                improved_vector, fitness = self._apply_kgls(vrp_solution)

                self._harmony_memory.append((improved_vector, fitness))

        # Sort solution memory
        self._harmony_memory.sort(key=lambda x: x[1], reverse=not self._obj_fun.maximize())

        # Store initial memory for history
        harmony_list = {'gen': 0, 'harmonies': self._harmony_memory.copy()}
        self._harmony_history = [harmony_list]

        logger.info("Harmony memory initialization complete")

    def run(self, initial_harmonies=None):
        """
        Run the hybrid HS-KGLS algorithm.

        Args:
            initial_harmonies: Optional list of initial harmonies

        Returns:
            Tuple of (best solution, best fitness, solution memory, solution history)
        """
        # Set random seed if specified
        if self._obj_fun.use_random_seed():
            random.seed(self._obj_fun.get_random_seed())

        # Initialize solution memory
        self._harmony_memory = []
        self._harmony_history = []
        self.initialize(initial_harmonies)

        # Main loop - create improvisations
        generation = 0
        num_imp = 0
        last_improvement_iteration = 0
        best_fitness = self._harmony_memory[0][1]

        logger.info(f"Starting main improvisation loop ({self._obj_fun.get_max_imp()} iterations)")

        while num_imp < self._obj_fun.get_max_imp():
            logger.info(f"Iteration {num_imp + 1}")
            # Generate new solution
            harmony = []

            # Compute new PAR if needed
            # self._obj_fun.compute_par(num_imp)
            num_discrete_values: int = 0
            # Apply standard HS operators
            for i in range(self._obj_fun.get_num_parameters()):
                num_discrete_values = self._obj_fun.get_num_discrete_values(i) - 1
                if random.random() <= self._obj_fun.get_hmcr():
                    self.memory_consideration(harmony, i)
                    if random.random() <= self._obj_fun.get_par():
                        self.pitch_adjustment(harmony, i)
                else:
                    if self._obj_fun.is_variable(i):
                        self.random_selection(harmony, i)
                    else:
                        harmony.append(0)
            logger.info(f"New solution generated: {harmony}.")

            # remove duplicates to make the solution valid for KGLS
            harmony = remove_duplicates(harmony, upper_bound=num_discrete_values)

            fitness = self._obj_fun.get_fitness(harmony)
            logger.info(f"Harmony after removing duplicates: {harmony}, Number of routes {len([x for x in harmony if x == 0])}, Fitness: {fitness}")

            # Convert to VRP solution, improve with KGLS and compute fitness
            vrp_solution = vector_to_vrp_solution(harmony, self._problem_instance)

            if 0 < fitness <= best_fitness:
                logger.info("Better solution found.")
                vrp_solution.to_file(fitness, self._write_to_file)
                best_fitness = fitness

            improved_harmony, fitness = self._apply_kgls(vrp_solution)

            # Update solution memory
            self.update_harmony_memory(improved_harmony, fitness)

            # Check if we found a better solution
            if ((self._obj_fun.maximize() and fitness > best_fitness) or
                    (not self._obj_fun.maximize() and fitness < best_fitness)):
                best_fitness = fitness
                last_improvement_iteration = num_imp
                logger.info(f"New best solution found at iteration {num_imp}: {best_fitness}")

            # Save harmonies every nth improvisations (one 'generation')
            if num_imp % self._obj_fun.get_hms() == 0:
                generation += 1
                harmony_list = {'gen': generation, 'harmonies': self._harmony_memory.copy()}
                self._harmony_history.append(harmony_list)
                logger.info(f"Completed generation {generation}")
                print("==========================================================")

            # Check if we should stop due to lack of improvement
            if num_imp - last_improvement_iteration > self._obj_fun.get_max_imp() // 5:
                logger.info(
                    f"Stopping early due to lack of improvement after {num_imp - last_improvement_iteration} iterations")
                break

            num_imp += 1

        # Return best solution
        best_harmony = None
        best_fitness = float('-inf') if self._obj_fun.maximize() else float('+inf')

        for harmony, fitness in self._harmony_memory:
            if ((self._obj_fun.maximize() and fitness > best_fitness) or
                    (not self._obj_fun.maximize() and fitness < best_fitness)):
                best_harmony = harmony
                best_fitness = fitness

        logger.info(f"Algorithm completed. Best solution: {best_harmony}. Fitness: {best_fitness}")

        return best_harmony, best_fitness, self._harmony_memory, self._harmony_history

    def random_selection(self, harmony, i):
        """
            Choose a random note from the possible values. Value assigned must not be 0, since 0 marks the end of route.
        """
        harmony.append(self._obj_fun.get_value(i, random.randint(1, self._obj_fun.get_num_discrete_values(i) - 1)))

    def pitch_adjustment(self, harmony, i):
        """
            If variable, randomly adjust the pitch up or down by some amount. The value assigned must not be 0, since 0 marks the end of route.
        """
        if self._obj_fun.is_variable(i):
            current_index = self._obj_fun.get_index(i, harmony[i])
            # discrete variable
            if random.random() < 0.5:
                # adjust pitch down
                new_value = self._obj_fun.get_value(i, current_index - random.randint(0,
                                                                                       min(self._obj_fun.get_mpai(),
                                                                                           current_index)))
                # if new value is 0, don't change
                harmony[i] = new_value if new_value != 0 else harmony[i]
            else:
                # adjust pitch up
                harmony[i] = self._obj_fun.get_value(i, current_index + random.randint(0,
                                                                                       min(self._obj_fun.get_mpai(),
                                                                                           self._obj_fun.get_num_discrete_values(
                                                                                               i) - current_index - 1)))
