from datastructure import VRPProblem, Node
from datastructure.vrp_objective_function import VRPObjectiveFunction, has_duplicates
from helpers.helpers import truncate_to_decimal, multiply_and_floor
from tests.unit_tests.test_local_search_operators.test_cross_exchange import build_toy_problem_vrptw


def build_toy_cvrp_problem():
    depot = Node(0, 38, 46, 0, True)
    customers = [
        Node(1, 59, 46, 16, False),
        Node(2, 96, 42, 18, False),
        Node(3, 47, 61, 1, False),
        Node(4, 26, 15, 13, False),
        Node(5, 66, 6, 8, False),
    ]
    nodes = [depot] + customers

    problem = VRPProblem('CVRP', nodes, 30)
    return problem

def test_has_duplicates():
    # Test the has_duplicates function
    assert has_duplicates([0, 1, 2, 3, 4, 5, 0]) == False
    assert has_duplicates([0, 1, 2, 3, 3, 5]) == True
    assert has_duplicates([]) == False
    assert has_duplicates([1]) == False
    assert has_duplicates([1, 1]) == True

class TestVRPObjectiveFunction:
    def test_init(self):
        arguments = {
            '--hms': 20,
            '--hmcr': 0.9,
            '--par': 0.5,
            '--ni': 1000,
        }

        problem_instance, cost_evaluator = build_toy_problem_vrptw()

        obj_func = VRPObjectiveFunction(arguments, problem_instance)

        assert obj_func._location_number == 7
        assert obj_func._vehicle_number == 3

        # Test with a different number of parameters
        obj_func = VRPObjectiveFunction(arguments, problem_instance, number_of_parameters=6)
        assert obj_func._number_of_parameters == 6

    def test_vrp_obj_func_kgls(self):
        arguments = {
            '--hms': 20,
            '--hmcr': 0.9,
            '--par': 0.5,
            '--ni': 1000,
        }

        problem_instance, cost_evaluator = build_toy_problem_vrptw()

        obj_func = VRPObjectiveFunction(arguments, problem_instance)

        assert obj_func._location_number == 7
        assert obj_func._vehicle_number == 3

        route = [3, 1, 2, 0, 0, 0, 0, 0]
        assert obj_func.get_fitness(route) == 63.4

        route = [6, 5, 4, 0, 0, 0, 0, 0]
        assert obj_func.get_fitness(route) == 90.1

        # try different variants of the route
        route = [0, 3, 1, 2, 0, 6, 5, 4]
        assert obj_func.get_fitness(route) == 153.5

        route = [3, 1, 2, 0, 6, 5, 4, 0]
        assert obj_func.get_fitness(route) == 153.5

        route = [3, 2, 0, 4, 4, 4, 4, 4]
        assert obj_func.get_fitness(route) == float('-inf')

        problem_instance = build_toy_cvrp_problem()

        obj_func = VRPObjectiveFunction(arguments, problem_instance, number_of_parameters=6)

        assert obj_func._location_number == 6
        assert obj_func._number_of_parameters == 6

        # Rounding error. Acceptable
        route = [3, 2, 5, 0, 1, 4]
        assert obj_func.get_fitness(route) == 265

        route = [1, 4, 0, 3, 2, 5]
        assert obj_func.get_fitness(route) == 265

def test_st():
    distance = 1350.0
    assert(multiply_and_floor(distance) == 13500)
    assert(truncate_to_decimal(distance) == 135.0)