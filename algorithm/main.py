import logging
import os
import sys
import argparse
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

    kgls = KGLS(arguments['<problem_instance>'], explicit_distances=True)
    # pm = PenaltyManager.init_from(problem_instance)

    max_runtime = int(arguments.get('--max_runtime', 10))
    kgls.set_abortion_condition("max_runtime", max_runtime)

    # generate first solution using KGLS
    kgls.run(visualize_progress=False)

    best_sol: str = kgls.best_solution.to_str(kgls.best_found_solution_value)

    return best_sol
    # print(f"Gap to BKS: {kgls.best_found_gap:.2f}")


def run_hs_kgls(arguments, logger):
    logger.info("Starting Hybrid HS-KGLS algorithm")

    solution_file = f'{arguments["<problem_instance>"].rsplit(".", 1)[0]}_sol.txt'

    # initialize problem
    problem_instance = read_vrp_instance(arguments['<problem_instance>'])

    # initialize guided local search algorithm
    kgls = KGLS(arguments['<problem_instance>'],
               neighborhood_size=len(problem_instance.nodes) if len(problem_instance.nodes) < 20 else 20)
    pm = PenaltyManager.init_from(problem_instance)
    kgls.cost_evaluator.set_infeasible_penalty_weights(pm.penalties[0], pm.penalties[1])

    max_runtime = int(arguments.get('--max_runtime', 100))
    kgls.set_abortion_condition("max_runtime", max_runtime)

    kgls.run(visualize_progress=False)

    kgls.best_solution_to_file(solution_file)
    print(f"Gap to BKS: {kgls.best_found_gap:.2f}")


def run_instance(arguments, logger):
    logger.info(f"Problem instance: {arguments['<problem_instance>']}")

    algorithm = arguments.get('--algorithm', 'h_kgls')

    if algorithm == 'hybrid_heuristic_hs':
        run_hybrid_heuristic_hs(arguments)
    elif algorithm == 'hs_kgls':
        run_hs_kgls(arguments, logger)
    else:  # Default to h_kgls
        return run_h_kgls(arguments, logger)


def run_benchmark(instance_dir, max_runtime=60):
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logger = logging.getLogger(__name__)
    instance_path = os.path.join(Path(__file__).resolve().parent, instance_dir)
    all_instances = sorted([f for f in os.listdir(instance_path) if f.endswith('.vrp')])[1:35]

    print(len(all_instances), "instances found")
    gaps = dict()
    run_times = dict()

    for file in all_instances:
        logger.info(f'Solving {file}')
        file_path = os.path.join(instance_path, file)

        # Let us use default parameters
        kgls = KGLS(file_path)
        kgls.set_abortion_condition("max_runtime", max_runtime)
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
    parser = argparse.ArgumentParser(description='Vehicle Routing Problem Solver')

    # Add arguments
    parser.add_argument('--instance', help='Path to the problem instance file', nargs='?')
    parser.add_argument('--algorithm', choices=['h_kgls', 'hs_kgls', 'hybrid_heuristic_hs'],
                        default='h_kgls', help='Algorithm to use (default: h_kgls)')
    parser.add_argument('--hms', type=int, default=20, help='Harmony memory size (default: 20)')
    parser.add_argument('--hmcr', type=float, default=0.9, help='Harmony memory consideration rate (default: 0.9)')
    parser.add_argument('--par', type=float, default=0.3, help='Pitch adjustment rate (default: 0.3)')
    parser.add_argument('--ni', type=int, default=1000, help='Number of improvisations (default: 1000)')
    parser.add_argument('--max_runtime', type=int, default=10, help='Maximum runtime in seconds (default: 300)')
    parser.add_argument('--benchmark', help='Run benchmark on specified instance directory')
    parser.add_argument('--benchmark_runtime', type=int, default=60,
                        help='Maximum runtime per instance in benchmark (default: 60)')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    # Handle benchmark mode
    if args.benchmark:
        run_benchmark(args.benchmark, args.benchmark_runtime)
        return

    # Ensure we have an instance file for normal operation
    if not args.instance:
        parser.error("Instance file path is required unless using --benchmark")

    # Convert args to the expected dictionary format
    arguments = {
        '<problem_instance>': args.instance,
        '--hms': str(args.hms),
        '--hmcr': str(args.hmcr),
        '--par': str(args.par),
        '--ni': str(args.ni),
        '--algorithm': args.algorithm,
        '--max_runtime': str(args.max_runtime)
    }
    print(arguments)
    print(run_instance(arguments, logger))


if __name__ == '__main__':
    freeze_support()
    main()