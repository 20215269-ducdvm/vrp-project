import logging
import math

from datastructure import Node

logger = logging.getLogger(__name__)


def remove_duplicates(solution, upper_bound: int, max_zeros_allowed: int):
    """
    Fix the format of the solution by replacing duplicate values with the nearest available value (except zero).
    """
    # logger.info(f"Removing duplicates from the current solution")

    # Use a dictionary to track seen values and their positions
    seen = {}

    num_zeros = 0
    for i in range(len(solution)):
        if solution[i] == 0:
            num_zeros += 1
            continue  # Skip zero values

        assert num_zeros <= max_zeros_allowed, "Too many routes in the solution"

        if solution[i] in seen:
            # This is a duplicate, find a replacement
            original = solution[i]
            offset = 1

            # Try alternating up and down until we find an unused value
            while True:
                # Try up
                up = original + offset
                if up <= upper_bound and up not in seen:
                    solution[i] = up
                    seen[up] = i
                    break

                # Try down
                down = original - offset
                if down >= 1 and down not in seen:
                    solution[i] = down
                    seen[down] = i
                    break

                offset += 1

                # If we've exhausted all possibilities, it means we can't find a valid replacement
                if original + offset > upper_bound and original - offset < 1:
                    solution[i] = 0
                    break
        else:
            seen[solution[i]] = i
    # logger.info(f"Solution after removing duplicates: {solution}")
    return solution

def multiply_and_floor(value, decimal_places=1):
    """Multiply by 10^decimal_places and floor the result"""
    factor = 10 ** decimal_places
    return math.floor(value * factor)


def truncate_to_decimal(value, decimal_places=1):
    """Truncate a value to a specific number of decimal places without rounding. Default is 1 decimal place."""
    factor = 10 ** decimal_places
    return value / factor


def compute_euclidean_distance(node1: Node, node2: Node) -> int:
    """
        Calculates Euclidean distance between two nodes.
        Returns int for CVRP problems (Node type) and float for VRPTW problems (NodeWithTW type).
    """
    distance = math.sqrt(
        math.pow(node1.x_coordinate - node2.x_coordinate, 2) +
        math.pow(node1.y_coordinate - node2.y_coordinate, 2)
    )
    return multiply_and_floor(distance, 1)
