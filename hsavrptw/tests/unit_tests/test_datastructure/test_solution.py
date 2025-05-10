import os

from pyharmonysearch.hybrid_heuristic_hs import decode_harmony_to_routes
from read_write import read_vrp_instance


def test_solution():
    from datastructure.vrp_objective_function import VRPObjectiveFunction, vrp_solution_to_vector
    from heuristics.kgls import KGLS
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    instance_path = os.path.join(root_dir, 'instances', 'X', 'X-n101-k25.vrp')

    arguments = {
        '_problem_instance': instance_path,
        '--hms': 20,
        '--hmcr': 0.9,
        '--par': 0.5,
        '--ni': 1000,
    }

    problem_instance = read_vrp_instance(arguments['_problem_instance'])

    kgls = KGLS(arguments['_problem_instance'], depth_lin_kernighan=1, depth_relocation_chain=1, num_perturbations=1,
                neighborhood_size=10)
    kgls.set_abortion_condition("runtime_without_improvement", 1)

    # initialize objective function for hybrid HS-GLS algorithm, containing VRP problem instance and parameters for HS algorithm
    obj_fun = VRPObjectiveFunction(arguments, problem_instance)

    # generate solution memory using Clarke-Wright heuristic
    # first_solution = clark_wright_route_reduction(vrp_instance=_problem_instance, cost_evaluator=kgls.cost_evaluator)
    kgls.run(visualize_progress=False)
    first_solution = kgls.best_solution
    fitness = obj_fun.get_fitness(vrp_solution_to_vector(first_solution))

def test_decode_harmony():
    harmony = [1, 2, 3, 0, 0, 4, 5, 6, 0]
    routes = decode_harmony_to_routes(harmony)
    assert routes == [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]]

