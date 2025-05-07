import math
import logging
from copy import copy

from datastructure import Node
from datastructure.node import NodeWithTW
from datastructure.route import RouteWithTW

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

def update_time_window(route: RouteWithTW):
    """
    Updates the time windows of the nodes in the route based on the arrival times.
    Args:
        route: The route to update
    """
    for i in range(len(route.nodes)):
        node = route.nodes[i]
        node.arrival_time = max(node.arrival_time, node.time_window[0])
        node.time_window = (node.arrival_time, node.time_window[1])

def create_route_with_tw(
        nodes: list[NodeWithTW],
        route_index: int
) -> RouteWithTW:
    """
    Creates a new route with time windows.
    Args:
        nodes: List of nodes to include in the route
        route_index: Index of the route

    Returns:
        A new RouteWithTW object
    """
    return RouteWithTW(nodes, route_index)

def create_modified_route(
        route: RouteWithTW,
        segment_to_remove: list[Node],
        segment_to_add: list[Node],
        segment_insert_after: Node
) -> RouteWithTW:
    """
    Creates a new route by removing one segment and adding another.
    Used to check time window constraints.
    Args:
        route: The route to modify
        segment_to_remove: List of nodes to remove from the route
        segment_to_add: List of nodes to add to the route
        segment_insert_after: Node after which to insert the new segment

    Returns:
        A new Route object with the modifications applied
    """
    # Create a copy of the original route
    new_route = copy(route)

    # Remove the segment to remove from the route
    for node in segment_to_remove:
        new_route.remove_customer(node)

    # Add the new segment to the route
    new_route.add_customers_after(segment_to_add, segment_insert_after)

    return new_route

