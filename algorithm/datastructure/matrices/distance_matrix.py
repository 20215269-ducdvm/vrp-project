import numpy as np

from datastructure import Node
from helpers.helpers import compute_euclidean_distance


class DistanceMatrix:
    _instances = {}

    def __new__(cls, nodes: list[Node]):
        """
        Creates a singleton instance of DistanceMatrix for a set of nodes.

        Parameters
        ----------
        nodes
            List of nodes to compute distances between
        """
        # Create a unique key based on node IDs and max_neighbors
        instance_key = tuple(node.node_id for node in nodes)

        # Return existing instance if available
        if instance_key in cls._instances:
            return cls._instances[instance_key]

        # Create new instance
        instance = super().__new__(cls)
        cls._instances[instance_key] = instance
        return instance

    def __init__(self, nodes: list[Node]):
        # Skip initialization if already initialized
        if hasattr(self, 'initialized'):
            return

        # Create mappings between node objects and indices
        self.nodes = nodes
        self.node_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}
        num_nodes = len(nodes)

        # Initialize NumPy arrays for costs
        self._distances = np.full((num_nodes, num_nodes), np.inf)

        # Compute costs as Euclidean distance between each pair of nodes
        for i, node1 in enumerate(nodes):
            # For depot nodes, compute distances to all nodes
            if node1.is_depot:
                for j, node2 in enumerate(nodes):
                    dist = compute_euclidean_distance(node1, node2)
                    self._distances[i, j] = dist
            else:
                # For non-depot nodes, find the closest nodes
                distances = []
                for j, node2 in enumerate(nodes):
                    if i != j:  # Skip the node itself
                        dist = compute_euclidean_distance(node1, node2)
                        if node2.is_depot:  # Always include depot
                            self._distances[i, j] = dist
                        else:
                            distances.append((dist, j))

                # Sort and select the closest neighbors
                distances.sort()  # Sort by distance
                for dist, j in distances:
                    self._distances[i, j] = dist

        self.initialized = True

    def get_distance(self, node1: Node, node2: Node) -> float:
        """
        Get the distance between two nodes.

        If the distance is not stored (infinity), it will be computed on the fly.
        """
        idx1 = self.node_to_idx[node1.node_id]
        idx2 = self.node_to_idx[node2.node_id]

        # Check if the distance is stored (not infinity)
        if np.isinf(self._distances[idx1, idx2]):
            # Calculate on the fly if not stored
            return compute_euclidean_distance(node1, node2)

        return self._distances[idx1, idx2]

    def get_matrix(self) -> np.ndarray:
        """
        Returns the full distance matrix.
        """
        return self._distances

    def get_nearest_neighbors(self, node: Node, count: int) -> list[Node]:
        """
        Get the nearest neighbors for a node.

        Parameters
        ----------
        node
            The node to find neighbors for
        count
            Maximum number of neighbors to return
        """
        idx = self.node_to_idx[node.node_id]

        # Get distances to all nodes
        distances = []
        for j, dist in enumerate(self._distances[idx]):
            if not np.isinf(dist) and j != idx:
                distances.append((dist, j))

        # Sort by distance
        distances.sort()

        # Return the nearest neighbors
        return [self.idx_to_node[j] for _, j in distances[:count]]
