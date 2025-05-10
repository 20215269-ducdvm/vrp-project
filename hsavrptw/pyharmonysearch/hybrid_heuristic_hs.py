"""
Enhanced Harmony Search for Vehicle Routing Problem with Time Windows (VRPTW).
This implementation adds the key components from the research paper:
1. Plug-in Heuristic Path Construction
2. Local Neighborhood Search

Following the exact flow in Figure 3 of the paper.
"""
import copy
import random
import logging

from datastructure import VRPProblem, CostEvaluator
from datastructure.route import compute_euclidean_distance
from helpers.helpers import remove_duplicates, create_route_with_tw
from pyharmonysearch.harmony_search import HarmonySearchVRP, insert_harmony_to_memory

logger = logging.getLogger(__name__)

def convert_note_to_node(note: int, problem_instance: VRPProblem):
    """
    Convert a note (customer index) to a node (customer object) using the problem instance.

    Args:
        note: Customer index
        problem_instance: The VRP problem instance

    Returns:
        Corresponding customer node
    """
    return problem_instance.nodes[note]
def convert_node_to_note(node, problem_instance: VRPProblem):
    """
    Convert a node (customer object) to a note (customer index) using the problem instance.

    Args:
        node: Customer node
        problem_instance: The VRP problem instance

    Returns:
        Corresponding customer index
    """
    return problem_instance.nodes.index(node)

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
    def __init__(self, obj_fun):
        """
        Initialize the Hybrid Heuristic Harmony Search Algorithm.

        Args:
            obj_fun: Objective function for the VRPTW
        """
        super().__init__(obj_fun)
        self.max_zeros_allowed = obj_fun.problem_instance.number_vehicles_required - 1

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

        # logger.info(f"Harmony Search algorithm initialized with random harmonies")
        # logger.info(f"Initial harmony memory: {self._harmony_memory}")
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
                        # Perturbation step (Step 7 in flowchart)
                        # This applies the heuristic operators for each component
                        # But we'll design it to act only when appropriate
                        self.apply_perturbation(harmony, i)
                else:
                    self.random_selection(harmony, i)

                # Note: 'i' is incremented by the loop itself (Step 8)
                # Then it checks if i reaches n (Step 4) in the for loop condition

            harmony = remove_duplicates(harmony, upper_bound=self._obj_fun.get_num_discrete_values(0) - 1,
                                        max_zeros_allowed=self._obj_fun.problem_instance.number_vehicles_required - 1)
            # After the solution is complete, calculate fitness
            fitness = self._obj_fun.get_fitness(harmony)
            # assert fitness >= 0

            self.update_harmony_memory(harmony, fitness)
            num_imp += 1

            # save harmonies every nth improvisations (i.e., one 'generation')
            if num_imp % self._obj_fun.get_hms() == 0:
                generation += 1
                harmony_list = {'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)}

                self._harmony_history.append(harmony_list)
        # return best solution
        best_harmony, best_fitness = self._return_best_harmony()

        logger.info(f"Best harmony found: {best_harmony}, fitness: {best_fitness}")
        return best_harmony, best_fitness, self._harmony_memory, self._harmony_history

    def _return_best_harmony(self):
        """
        Return the best harmony from the memory.

        This is a helper function to retrieve the best harmony after the search process.

        Returns:
            The best harmony and its fitness value
        """
        best_harmony = None
        best_fitness = float('+inf')
        for harmony, fitness in self._harmony_memory:
            if (self._obj_fun.maximize() and fitness > best_fitness) or (
                    not self._obj_fun.maximize() and fitness < best_fitness):
                best_harmony = harmony
                best_fitness = fitness

        return best_harmony, best_fitness

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
                    harmony = remove_duplicates(harmony, upper_bound=self._obj_fun.get_num_discrete_values(0) - 1, max_zeros_allowed=self.max_zeros_allowed)
                    fitness = self._obj_fun.get_fitness(harmony)
                    if fitness > float('-inf'):
                        break
                initial_harmonies = insert_harmony_to_memory(initial_harmonies, harmony, fitness)

        self._harmony_memory = initial_harmonies

        harmony_list = {'gen': 0, 'harmonies': copy.deepcopy(self._harmony_memory)}
        self._harmony_history.append(harmony_list)

    def random_selection(self, harmony, i):
        """
        Randomly choose a value for the current component.

        This is Step 5 in the flowchart, where we choose a random note.

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

        This is Step 6 in the flowchart, where we consider a note from the harmony memory.

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
    def apply_perturbation(self, harmony, i):
        """
        Apply perturbation to the solution after each component is added.

        Following the flowchart in the paper, this is Step 7, occurring after
        pitch adjustment and before incrementing i.

        Args:
            harmony: Current (partial) solution
            i: Current component index
        """
        # print(f"Perturbation called with solution length: {len(harmony)}, i: {i}")
        # Track if this is the last component (when i == n-1)
        is_last_component = (i == self._obj_fun.get_num_parameters() - 1)

        # For partial harmonies, we can either:
        # 1. Do nothing (skip perturbation)
        # 2. Apply a simplified perturbation appropriate for partial harmonies
        # 3. Apply full perturbation only on the last component

        if len(harmony) < 2 and not is_last_component:
            # Skip perturbation for very small partial harmonies
            # logger.info(f"Skipping perturbation for partial solution with {len(harmony)} elements")
            return harmony

        if is_last_component:
            # Full perturbation on the last component
            # logger.info(f"Applying full perturbation on complete solution: {harmony}")
            harmony = remove_duplicates(harmony, upper_bound=self._obj_fun.get_num_discrete_values(i) - 1, max_zeros_allowed=self._obj_fun.problem_instance.number_vehicles_required - 1)
            harmony = self.plug_in_heuristic_path_construction(harmony)
            harmony = self.local_neighborhood_search(harmony)
        else:
            # A simplified perturbation for partial harmonies
            # This could be a lighter version appropriate for partial solutions
            # logger.info(f"Applying simplified perturbation on partial solution: {harmony}")
            # For example, just remove duplicates to assure that the solution contains all customers
            harmony = remove_duplicates(harmony, upper_bound=self._obj_fun.get_num_discrete_values(i) - 1, max_zeros_allowed=self._obj_fun.problem_instance.number_vehicles_required - 1)


        return harmony

    def plug_in_heuristic_path_construction(self, harmony: list[int]) -> list[int]:
        """
        Implement the plug-in heuristic path construction as described in Section IV.D.

        This method adjusts the generated path to ensure time window constraints are met.

        Args:
            harmony: Original solution (route)

        Returns:
            Improved solution with time window constraints satisfied
        """
        # Safety check - if solution is too small for a valid route, return as is
        if len(harmony) < 2:
            return harmony

        # Decode solution into routes
        routes = decode_harmony_to_routes(harmony)

        improved_routes = []
        for route in routes:
            if len(route) <= 2:  # Only contains depot (start and end)
                continue

            # Step 1: Randomly select a seed customer
            seed_idx = random.randint(1, len(route) - 2)  # Exclude depot
            seed_customer = route[seed_idx]

            # Step 2: Form initial path with the seed customer
            new_route = [0, seed_customer, 0]  # Depot - Seed - Depot
            remaining_customers = [c for i, c in enumerate(route[1:-1]) if i + 1 != seed_idx]

            # Step 3: Insert remaining customers at appropriate positions
            for customer in remaining_customers:
                best_pos = None
                best_cost = float('inf')

                # Try inserting at each position
                for i in range(1, len(new_route)):
                    # Check if insertion is feasible (TW constraints)
                    temp_route = new_route.copy()
                    temp_route.insert(i, customer)

                    if self._is_tw_feasible(temp_route):
                        # Calculate cost (e.g., added distance)
                        cost = self._calculate_insertion_cost(temp_route, i)
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = i

                # Insert at best position if found
                if best_pos is not None:
                    new_route.insert(best_pos, customer)
                else:
                    # If no feasible position found, add a new route
                    new_route = [0, customer, 0]

            improved_routes.append(new_route)

        # Convert improved routes back to a single solution
        improved_harmony = encode_routes_to_harmony(improved_routes)
        return improved_harmony

    def local_neighborhood_search(self, harmony: list[int]) -> list[int]:
        """
        Implement local neighborhood search as described in Section IV.E.

        This includes:
        1. Correlation Deletion Operator
        2. Feasible Optimal Reinsertion Operator

        Args:
            harmony: Original solution (route)

        Returns:
            Improved solution after local search
        """
        # Safety check - if solution is too small for a valid route, return as is
        if len(harmony) < 2:
            return harmony

        # Decode solution into routes
        routes = decode_harmony_to_routes(harmony)

        # Ensure we have at least one non-empty route
        valid_routes = [route for route in routes if len(route) > 2]
        if not valid_routes:
            return harmony  # Return original solution if no valid routes

        # Step 1: Correlation Deletion Operator
        # Determine number of customers to delete (10-20% of total)
        total_customers = self._obj_fun.location_number - 1 # Exclude depots
        if total_customers == 0:
            return harmony  # Return original solution if no customers

        num_to_delete = max(1, int(random.uniform(0.1, 0.2) * total_customers))

        # Randomly select a customer
        random_route_idx = random.randint(0, len(valid_routes) - 1)
        random_route = valid_routes[random_route_idx]
        random_customer_idx = random.randint(1, len(random_route) - 2)
        random_customer = random_route[random_customer_idx]

        # Calculate correlation with all customers
        correlations = []
        for r_idx, route in enumerate(routes):
            for c_idx, customer in enumerate(route):
                if c_idx == 0 or c_idx == len(route) - 1:  # Skip depot
                    continue

                # Calculate correlation as defined in the paper
                dist = self._calculate_distance(random_customer, customer)
                max_dist = self._obj_fun.get_max_distance(customer, neighborhood_size=total_customers)

                # Avoid division by zero
                c = dist / max_dist if max_dist > 0 else 0

                # Check if on the same route
                r_orig_idx = routes.index(random_route)
                v = 1 if r_idx != r_orig_idx else 0

                # Avoid division by zero
                denominator = c + v
                correlation = 1 / denominator if denominator > 0 else float('inf')

                correlations.append((correlation, r_idx, c_idx, customer))

        # Sort by correlation and select the customers with the highest correlation
        correlations.sort(reverse=True)
        customers_to_remove = correlations[:num_to_delete]

        # Remove selected customers from routes
        removed_customers = []
        for _, r_idx, c_idx, customer in customers_to_remove:
            if 0 <= r_idx < len(routes) and 0 < c_idx < len(routes[r_idx]) - 1:  # Safety check
                if routes[r_idx][c_idx] == customer:  # Make sure it's the right customer
                    routes[r_idx].pop(c_idx)
                    removed_customers.append(customer)

        # Step 2: Feasible Optimal Reinsertion Operator
        for customer in removed_customers:
            best_route_idx = -1
            best_position = -1
            best_cost = float('inf')

            # Try inserting in all possible positions
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    # Check if insertion is feasible (capacity and TW constraints)
                    temp_route = route.copy()
                    temp_route.insert(pos, customer)

                    if self._is_feasible(temp_route):
                        # Calculate cost (e.g., added distance)
                        cost = self._calculate_insertion_cost(temp_route, pos)
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = r_idx
                            best_position = pos

            # Insert at best position if found
            if best_route_idx != -1:
                routes[best_route_idx].insert(best_position, customer)
            else:
                # If no feasible position found, create a new route
                routes.append([0, customer, 0])

        # Convert improved routes back to a single solution
        improved_harmony = encode_routes_to_harmony(routes)
        return improved_harmony

    def _is_tw_feasible(self, nodes: list[int]) -> bool:
        """
        Check if a route satisfies time window constraints.

        Args:
            nodes: A sequence of customer indices

        Returns:
            True if the route is feasible, False otherwise
        """
        nodes_list = [self._obj_fun.problem_instance.nodes[i] for i in nodes]
        route = create_route_with_tw(nodes_list, -1)
        return route.can_update_time_windows()

    def _is_feasible(self, route: list[int]) -> bool:
        """
        Check if a route is feasible (capacity and time window constraints).

        Args:
            route: A sequence of customer indices

        Returns:
            True if the route is feasible, False otherwise
        """
        # Check capacity constraints
        vehicle_capacity = self._obj_fun.problem_instance.capacity
        total_demand = sum(self._obj_fun.problem_instance.nodes[i].demand for i in route)
        capacity_check = total_demand <= vehicle_capacity

        time_window_check = True
        # Check time window constraints
        if self._obj_fun.problem_instance.type == "VRPTW":
            time_window_check = self._is_tw_feasible(route)

        return capacity_check and time_window_check

    def _calculate_insertion_cost(self, route: list[int], pos: int) -> float:
        """
        Calculate the cost of inserting a customer at a specific position.

        Args:
            route: The route after insertion
            pos: Position where the customer was inserted

        Returns:
            Cost of insertion (e.g., added distance)
        """
        # For simplicity, calculate the added distance

        if pos <= 0 or pos >= len(route):
            return float('inf')

        # Calculate delta distance: d(prev, new) + d(new, next) - d(prev, next)
        prev_node = route[pos - 1]
        new_node = route[pos]
        next_node = route[pos + 1]

        cost = (self._calculate_distance(prev_node, new_node) +
                self._calculate_distance(new_node, next_node) -
                self._calculate_distance(prev_node, next_node))

        return cost

    def _calculate_distance(self, node1: int, node2: int) -> float:
        """
        Calculate the Euclidean distance between two nodes.

        Args:
            node1: First node index
            node2: Second node index

        Returns:
            Euclidean distance between the nodes
        """
        # This should be implemented based on your specific VRPTW problem
        # and the distance calculation in your objective function

        node1 = self._obj_fun.problem_instance.nodes[node1]
        node2 = self._obj_fun.problem_instance.nodes[node2]
        return compute_euclidean_distance(node1, node2)
