import numpy as np
import re
from datastructure import Node
from helpers.helpers import compute_euclidean_distance


class DistanceMatrix:
    _instances = {}

    def __new__(cls, nodes=None, file_path=None):
        """
        Creates a singleton instance of DistanceMatrix for a set of nodes or from a file.
        """
        if file_path:
            # Create a unique key based on the file path
            instance_key = file_path
        elif nodes:
            # Create a unique key based on node IDs
            instance_key = tuple(node.node_id for node in nodes)
        else:
            raise ValueError("Either nodes or file_path must be provided")

        # Return existing instance if available
        if instance_key in cls._instances:
            return cls._instances[instance_key]

        # Create new instance
        instance = super().__new__(cls)
        cls._instances[instance_key] = instance
        return instance

    def __init__(self, nodes=None, file_path=None):
        # Skip initialization if already initialized
        if hasattr(self, 'initialized'):
            return

        if file_path:
            self._init_from_file(file_path)
        elif nodes:
            self._init_from_nodes(nodes)
        else:
            raise ValueError("Either nodes or file_path must be provided")

        self.initialized = True

    def _init_from_nodes(self, nodes):
        """Initialize distance matrix from a list of nodes using Euclidean distance."""
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
                # For non-depot nodes
                for j, node2 in enumerate(nodes):
                    if i != j:  # Skip the node itself
                        dist = compute_euclidean_distance(node1, node2)
                        self._distances[i, j] = dist

    def _init_from_file(self, file_path):
        """Initialize distance matrix from a VRPLIB file with FULL_MATRIX format."""
        # Parse file to get dimension and distance data
        dimension = 0
        edge_weight_format = ""
        edge_weight_section = False
        distances = []
        nodes_coords = []
        demands = []
        depot_idx = None

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                if line.startswith('DIMENSION'):
                    dimension = int(line.split(':')[1].strip())

                elif line.startswith('EDGE_WEIGHT_FORMAT'):
                    edge_weight_format = line.split(':')[1].strip()

                elif line == 'EDGE_WEIGHT_SECTION':
                    edge_weight_section = True
                    continue

                elif edge_weight_section and not (line == 'NODE_COORD_SECTION' or
                                                  line == 'DEMAND_SECTION' or
                                                  line == 'DEPOT_SECTION'):
                    # Parse distance values until we reach the next section
                    values = [float(val) for val in line.split()]
                    distances.extend(values)

                elif line == 'NODE_COORD_SECTION':
                    edge_weight_section = False
                    continue

                elif not edge_weight_section and re.match(r'^\d+\s+\d+\s+\d+$', line):
                    # Parse node coordinates
                    node_id, x, y = map(float, line.split())
                    nodes_coords.append((int(node_id), x, y))

                elif line == 'DEMAND_SECTION':
                    continue

                elif re.match(r'^\d+\s+\d+$', line):
                    # Parse demands
                    node_id, demand = map(int, line.split())
                    demands.append((node_id, demand))

                elif line == 'DEPOT_SECTION':
                    continue

                elif re.match(r'^\d+$', line) and line != '-1':
                    # Parse depot index
                    depot_idx = int(line)

        # Create distance matrix
        self._distances = np.zeros((dimension, dimension))

        if edge_weight_format == 'FULL_MATRIX':
            # Fill the matrix
            idx = 0
            for i in range(dimension):
                for j in range(dimension):
                    self._distances[i, j] = distances[idx]
                    idx += 1

        # Create Node objects
        self.nodes = []
        for node_id, x, y in nodes_coords:
            is_depot = (node_id == depot_idx)
            demand = 0
            for d_id, d_val in demands:
                if d_id == node_id:
                    demand = d_val
                    break

            # Create a Node object (assuming Node class constructor)
            node = Node(node_id=node_id, x_coordinate=x, y_coordinate=y, demand=demand, is_depot=is_depot)
            self.nodes.append(node)

        # Create mappings
        self.node_to_idx = {node.node_id: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}

    @classmethod
    def init_from_file(cls, file_path):
        """Initialize a DistanceMatrix from a VRPLIB file."""
        return cls(file_path=file_path)

    def get_distance(self, node1, node2):
        """Get the distance between two nodes."""
        idx1 = self.node_to_idx[node1.node_id]
        idx2 = self.node_to_idx[node2.node_id]
        return self._distances[idx1, idx2]

    def get_matrix(self):
        """Returns the full distance matrix."""
        return self._distances

    def get_nearest_neighbors(self, node, count):
        """Get the nearest neighbors for a node."""
        idx = self.node_to_idx[node.node_id]
        distances = [(self._distances[idx, j], j) for j in range(len(self.nodes)) if j != idx]
        distances.sort()
        return [self.idx_to_node[j] for _, j in distances[:count]]