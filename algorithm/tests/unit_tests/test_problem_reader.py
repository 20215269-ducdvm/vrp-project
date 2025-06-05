import os

from read_write import read_vrp_instance


def test_problem_reader():
    # Get the path to the root directory of the project
    # Assuming tests/unit_tests is 2 levels deep from the root
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    instance_path = os.path.join(root_dir, 'instances', 'Solomon', 'toy.txt')

    arguments = {
        '_problem_instance': instance_path,
        '--hms': 20,
        '--hmcr': 0.9,
        '--par': 0.5,
        '--ni': 1000,
    }

    problem_instance = read_vrp_instance(arguments['_problem_instance'])

    assert problem_instance is not None
    assert problem_instance.type == 'VRPTW'
    assert len(problem_instance.nodes) == 7
    assert problem_instance.nodes[0].x_coordinate == 50
    assert problem_instance.nodes[4].demand == 20
    assert problem_instance.nodes[4].time_window[1] == 100

    instance_path = os.path.join(root_dir, 'instances', 'X', 'toy.vrp')

    arguments = {
        '_problem_instance': instance_path,
        'hms': 20,
        'hmcr': 0.9,
        'par': 0.5,
        'ni': 1000,
    }

    problem_instance = read_vrp_instance(arguments['_problem_instance'])

    assert problem_instance is not None
    assert problem_instance.type == 'CVRP'
    assert len(problem_instance.nodes) == 6
    assert problem_instance.nodes[0].x_coordinate == 38
