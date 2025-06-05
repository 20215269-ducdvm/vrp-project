import numpy as np

from datastructure import Node
from helpers.helpers import compute_euclidean_distance


class DurationCalculator:
    """
    Base class for duration calculation strategies.
    """

    def calculate(self, node1: Node, node2: Node) -> float:
        """
        Calculate duration between two nodes.

        Parameters
        ----------
        node1
            The first node
        node2
            The second node

        Returns
        -------
        float
            Duration between the two nodes
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class EuclideanDurationCalculator(DurationCalculator):
    def calculate(self, node1: Node, node2: Node) -> float:
        return compute_euclidean_distance(node1, node2)


class ManhattanDurationCalculator(DurationCalculator):
    def calculate(self, node1: Node, node2: Node) -> float:
        return abs(node1.x_coordinate - node2.x_coordinate) + abs(node1.y_coordinate - node2.y_coordinate)


class CustomDurationCalculator(DurationCalculator):
    def __init__(self, duration_fn):
        self._duration_fn = duration_fn

    def calculate(self, node1: Node, node2: Node) -> float:
        return self._duration_fn(node1, node2)


class DurationMatrix:
    _instances = {}

    def __new__(cls, nodes: list[Node], calculator: DurationCalculator = None):
        """
        Creates a singleton instance of DurationMatrix for a set of nodes.

        Parameters
        ----------
        nodes
            List of nodes to compute durations between
        calculator
            Duration calculator strategy to use
        """
        # Create a unique key based on node IDs, calculator type, and max_neighbors
        calculator_type = type(calculator).__name__ if calculator else "None"
        instance_key = (tuple(node.node_id for node in nodes), calculator_type)

        # Return existing instance if available
        if instance_key in cls._instances:
            return cls._instances[instance_key]

        # Create new instance
        instance = super().__new__(cls)
        cls._instances[instance_key] = instance
        return instance

    def __init__(self, nodes: list[Node], calculator: DurationCalculator = None):
        # Skip initialization if already initialized
        if hasattr(self, 'initialized'):
            return

        # Create mappings between node objects and indices
        self.nodes = nodes
        self.node_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}
        num_nodes = len(nodes)

        # Initialize NumPy arrays for durations
        self._durations = np.full((num_nodes, num_nodes), np.inf)

        # Set default calculator if none provided
        self._calculator = calculator if calculator else EuclideanDurationCalculator()

        # Initialize durations for specific node pairs
        self._initialize_durations()

        self.initialized = True

    def _initialize_durations(self):
        """
        Initialize durations based on node types.
        """
        for i, node1 in enumerate(self.nodes):
            # For depot nodes, compute durations to all nodes
            if node1.is_depot:
                for j, node2 in enumerate(self.nodes):
                    duration = self._calculator.calculate(node1, node2)
                    self._durations[i, j] = duration
            else:
                # For non-depot nodes, find the closest nodes
                durations = []
                for j, node2 in enumerate(self.nodes):
                    if i != j:  # Skip the node itself
                        duration = self._calculator.calculate(node1, node2)
                        if node2.is_depot:  # Always include depot
                            self._durations[i, j] = duration
                        else:
                            durations.append((duration, j))

                # Sort and select the closest neighbors
                durations.sort()  # Sort by duration
                for duration, j in durations:
                    self._durations[i, j] = duration

    def set_calculator(self, calculator: DurationCalculator):
        """
        Sets a new calculator strategy for duration calculations.
        This will clear the singleton instance and create a new one.

        Parameters
        ----------
        calculator
            The new duration calculator strategy
        """
        # Find and remove the current instance from the cache
        keys_to_remove = []
        for key in self.__class__._instances:
            if self.__class__._instances[key] is self:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.__class__._instances[key]

        # Update calculator and re-initialize durations
        self._calculator = calculator
        self._durations = np.full((len(self.nodes), len(self.nodes)), np.inf)
        self._initialize_durations()

    def get_duration(self, node1: Node, node2: Node) -> float:
        """
        Get the duration between two nodes.

        If the duration is not stored (infinity), it will be computed on the fly.
        """
        idx1 = self.node_to_idx[node1.node_id]
        idx2 = self.node_to_idx[node2.node_id]

        # Check if the duration is stored (not infinity)
        if np.isinf(self._durations[idx1, idx2]):
            # Calculate on the fly if not stored
            duration = self._calculator.calculate(node1, node2)
            self._durations[idx1, idx2] = duration
            return duration

        return self._durations[idx1, idx2]

    def get_matrix(self) -> np.ndarray:
        """
        Returns the full duration matrix.
        """
        return self._durations.copy()

    def get_nearest_neighbors(self, node: Node, count: int) -> list[Node]:
        """
        Get the nearest neighbors for a node based on duration.

        Parameters
        ----------
        node
            The node to find neighbors for
        count
            Maximum number of neighbors to return
        """
        idx = self.node_to_idx[node.node_id]

        # Get durations to all nodes
        durations = []
        for j, duration in enumerate(self._durations[idx]):
            if not np.isinf(duration) and j != idx:
                durations.append((duration, j))

        # Sort by duration
        durations.sort()

        # Return the nearest neighbors
        return [self.idx_to_node[j] for _, j in durations[:count]]

    def compute_full_matrix(self):
        """
        Computes the full duration matrix for all node pairs.
        """
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if np.isinf(self._durations[i, j]):
                    self._durations[i, j] = self._calculator.calculate(node1, node2)


# Convenience function to create a DurationMatrix with specific calculator
def create_duration_matrix(nodes: list[Node], calculation_method: str = "euclidean",
                           ) -> DurationMatrix:
    """
    Creates a duration matrix with the specified calculation method.

    Parameters
    ----------
    nodes
        List of nodes to compute durations between
    calculation_method
        Method to use for duration calculation. Default is "euclidean"
        Available methods: "euclidean", "manhattan"
    Returns
    -------
    DurationMatrix
        A duration matrix with the specified calculator
    """
    if calculation_method not in ["euclidean", "manhattan"]:
        raise ValueError(f"Unsupported calculation method: {calculation_method}. Create a new custom method if you intend to use it.")

    calculators = {
        "euclidean": EuclideanDurationCalculator(),
        "manhattan": ManhattanDurationCalculator(),
    }

    calculator = calculators.get(calculation_method, EuclideanDurationCalculator())
    return DurationMatrix(nodes, calculator)