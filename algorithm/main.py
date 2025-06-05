import logging
import os
import sys
from multiprocessing import cpu_count, freeze_support
from pathlib import Path

from datastructure.penalty_manager import PenaltyManager
from datastructure.vrp_objective_function import VRPObjectiveFunction
from heuristics.kgls import KGLS
from pyharmonysearch.harmony_search import harmony_search, harmony_search_serial
from pyharmonysearch.hybrid_hs import HybridHeuristicHSA, vrp_solution_to_vector
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
    num_iterations = 50

    results = (harmony_search(obj_fun, num_processes, num_iterations, algorithm))
    # results = (harmony_search_serial(obj_fun, num_iterations, algorithm))
    best_harmony = results.best_harmony
    print("Time elapsed:", results.elapsed_time)
    print("Best solution:", best_harmony)
    print("Best fitness:", results.best_fitness)

def run_h_kgls(arguments, logger):
    logger.info("Starting Hybrid HS-KGLS algorithm")

    solution_file = f'{arguments["<problem_instance>"].rsplit(".", 1)[0]}_sol.txt'

    # initialize problem
    problem_instance = read_vrp_instance(arguments['<problem_instance>'])

    # initialize hybrid hs - guided local search algorithm

    kgls = KGLS(arguments['<problem_instance>'])
    pm = PenaltyManager.init_from(problem_instance)

    kgls.set_abortion_condition("max_runtime", 300)

    # generate first solution using KGLS
    kgls.run(visualize_progress=False)

    kgls.best_solution_to_file(solution_file)
    # print(f"Gap to BKS: {kgls.best_found_gap:.2f}")

def run_hs_kgls(arguments, logger):
    logger.info("Starting Hybrid HS-KGLS algorithm")

    solution_file = f'{arguments["<problem_instance>"].rsplit(".", 1)[0]}_sol.txt'

    # initialize problem
    problem_instance = read_vrp_instance(arguments['<problem_instance>'])

    # initialize guided local search algorithm
    kgls = KGLS(arguments['<problem_instance>'])
    pm = PenaltyManager.init_from(problem_instance)
    kgls.cost_evaluator.set_infeasible_penalty_weights(pm.penalties[0], pm.penalties[1])
    kgls.set_abortion_condition("max_runtime", 100)

    kgls.run(visualize_progress=False)

    kgls.best_solution_to_file(solution_file)
    print(f"Gap to BKS: {kgls.best_found_gap:.2f}")

def run_instance():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logger = logging.getLogger(__name__)

    dataset = "X/X-n936-k151.vrp"  # Example dataset
    # dataset = "X/X-n101-k25.vrp"  # Example dataset
    # dataset = "Antwerp/Antwerp1.vrp"
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
    # run_hybrid_heuristic_hs(arguments)
    run_h_kgls(arguments, logger)

def run_benchmark():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logger = logging.getLogger(__name__)
    instance_path = os.path.join(Path(__file__).resolve().parent, 'instances/X')
    all_instances = sorted([f for f in os.listdir(instance_path) if f.endswith('.vrp')])[1:35]

    print(len(all_instances), "instances found")
    gaps = dict()
    run_times = dict()

    for file in all_instances:
        logger.info(f'Solving {file}')
        file_path = os.path.join(instance_path, file)

        # Let us use default parameters
        kgls = KGLS(file_path)
        kgls.set_abortion_condition("max_runtime", 60)
        kgls.run(visualize_progress=False)

        # collects run stats
        kgls.print_time_distribution()
        gaps[file] = kgls.best_found_gap
        run_times[file] = kgls.total_runtime

    logger.info(f'Benchmark summary')
    logger.info(f'Average gap: {sum(gaps.values()) / len(gaps):.2f}')
    logger.info(f'Average run_time: {sum(run_times.values()) / len(run_times):.0f}')
    logger.info(f'Detailed Results')
    logger.info(f"{'Instance':<20}{'Time':<5}{'Gap':<5}")
    logger.info("-" * 30)
    for instance in gaps.keys():
        logger.info(f"{instance:<20}{int(run_times[instance]):<5}{gaps[instance]:.2f}")

def main():
    # run_benchmark()
    run_instance()

if __name__ == '__main__':
    freeze_support()
    main()