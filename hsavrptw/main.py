import logging
from multiprocessing import cpu_count, freeze_support

from constraint_checker import solution_to_routes
from datastructure.vrp_objective_function import VRPObjectiveFunction, vrp_solution_to_vector
from heuristics.guided_local_search.kgls import KGLS
from pyharmonysearch.harmony_search import harmony_search, harmony_search_serial
from pyharmonysearch.hybrid_heuristic_hs import HybridHeuristicHSA
from pyharmonysearch.hybrid_hs_kgls import HybridHSKGLS
from read_write import read_vrp_instance


def run_hybrid_heuristic_hs(arguments):
    """
    Run the Hybrid Harmony Search algorithm for Vehicle Routing Problem (VRP).
    """
    problem_instance = read_vrp_instance(arguments['<problem_instance>'])
    obj_fun = VRPObjectiveFunction(arguments, problem_instance)
    algorithm = HybridHeuristicHSA(obj_fun)
    num_processes = cpu_count() - 1  # use number of logical CPUs - 1 so that I have one available for use
    num_iterations = 100

    results = (harmony_search(obj_fun, num_processes, num_iterations, algorithm))
    # results = (harmony_search_serial(obj_fun, num_iterations, algorithm))
    best_harmony = results.best_harmony
    print("Time elapsed:", results.elapsed_time)
    print("Best solution:", best_harmony)
    print("Best fitness:", results.best_fitness)

def run_hs_kgls(arguments, logger):
    logger.info("Starting Hybrid HS-KGLS algorithm")

    solution_file = f'{arguments["<problem_instance>"].rsplit(".", 1)[0]}_sol.txt'

    # initialize problem
    problem_instance = read_vrp_instance(arguments['<problem_instance>'])

    # initialize guided local search algorithm
    kgls = KGLS(arguments['<problem_instance>'], depth_lin_kernighan=1,
                depth_relocation_chain=1, num_perturbations=1, neighborhood_size=10)
    kgls.set_abortion_condition("runtime_without_improvement", 1)

    # generate first solution using KGLS
    kgls.run(visualize_progress=False)
    first_solution = kgls.best_solution

    # convert solution to vector representation
    curr_solution = vrp_solution_to_vector(first_solution)

    # initialize objective function
    obj_fun = VRPObjectiveFunction(arguments, problem_instance, number_of_parameters=len(curr_solution),
                                   initial_solution=curr_solution)
    fitness = obj_fun.get_fitness(curr_solution)

    logger.info(
        f"First solution generated using KGLS: {curr_solution}, Fitness: {fitness}, Number of routes: {len([x for x in curr_solution if x == 0])}")
    first_solution.to_file(fitness, solution_file)

    # initialize solution memory
    initial_hm = []
    for i in range(int(arguments['--hms'])):
        initial_hm.append((curr_solution, fitness))

    # initialize solution search algorithm
    algorithm = HybridHSKGLS(obj_fun, kgls, solution_file)

    num_processes = cpu_count() - 1
    num_iterations = 10

    # run algorithm
    # results = harmony_search(obj_fun, num_processes, num_iterations, algorithm, initial_harmonies=initial_hm)
    results = harmony_search_serial(obj_fun, num_iterations, algorithm, initial_harmonies=initial_hm)

    best_harmony = results.best_harmony
    print("Time elapsed:", results.elapsed_time)
    print("Best solution:", best_harmony)
    print("Best fitness:", results.best_fitness)
    print("Route:", solution_to_routes(best_harmony))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    dataset = "Solomon/R101.txt"
    instance_path = f"instances/{dataset}"

    arguments = {
        '<problem_instance>': instance_path,
        '--hms': '20',  # Harmony memory size
        '--hmcr': '0.9',  # Harmony memory consideration rate
        '--par': '0.3',  # Pitch adjustment rate
        '--ni': '1000'  # Number of improvisations
    }

    logger.info(f"Problem instance: {arguments['<problem_instance>']}")

    # Uncomment one of these to run
    # run_hs(arguments)
    run_hybrid_heuristic_hs(arguments)
    # run_hs_kgls(arguments, logger)


if __name__ == '__main__':
    freeze_support()
    main()