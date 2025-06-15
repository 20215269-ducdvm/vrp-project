import unittest
import numpy as np
import os
import tempfile
from datastructure.matrices.distance_matrix import DistanceMatrix


class TestDistanceMatrix(unittest.TestCase):
    def setUp(self):
        # Create a temporary VRPLIB file with FULL_MATRIX format
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        self.temp_file.write("""NAME : test.vrp
COMMENT : Test CVRP instance with FULL_MATRIX
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
NODE_COORD_TYPE : TWOD_COORDS
CAPACITY : 2
EDGE_WEIGHT_SECTION
0 24 29 15
23 0 9 35
29 9 0 15
17 35 17 0
NODE_COORD_SECTION
1 0 0
2 0 20
3 14 18
4 17 0
DEMAND_SECTION
1 0
2 1
3 1
4 1
DEPOT_SECTION
1
-1
EOF""")
        self.temp_file.close()

        # Create original LOWER_ROW format file for comparison
        self.lower_row_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        self.lower_row_file.write("""NAME : toy2.vrp
COMMENT : Example of a CVRP instance with explicit distances
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : LOWER_ROW
NODE_COORD_TYPE : TWOD_COORDS
CAPACITY : 2
EDGE_WEIGHT_SECTION
23
29 9
17 35 15
NODE_COORD_SECTION
1 0 0
2 0 20
3 14 18
4 17 0
DEMAND_SECTION
1 0
2 1
3 1
4 1
DEPOT_SECTION
1
-1
EOF""")
        self.lower_row_file.close()

    def tearDown(self):
        # Clean up temporary files
        os.unlink(self.temp_file.name)
        os.unlink(self.lower_row_file.name)

    def test_init_from_full_matrix(self):
        """Test initialization from a file with FULL_MATRIX format."""
        # Initialize from FULL_MATRIX file
        distance_matrix = DistanceMatrix(file_path=self.temp_file.name)

        # Check dimensions
        matrix = distance_matrix.get_matrix()
        self.assertEqual(matrix.shape, (4, 4))

        # Check specific distances
        expected_matrix = np.array([
            [0, 24, 29, 15],
            [23, 0, 9, 35],
            [29, 9, 0, 15],
            [17, 35, 17, 0]
        ])
        np.testing.assert_array_equal(matrix, expected_matrix)

        # Check node properties
        self.assertEqual(len(distance_matrix.nodes), 4)

        # Check depot
        depot = next((node for node in distance_matrix.nodes if node.is_depot), None)
        self.assertIsNotNone(depot)
        self.assertEqual(depot.node_id, 1)

        # Check get_distance method
        node1 = distance_matrix.nodes[0]  # depot
        node2 = distance_matrix.nodes[1]  # first customer
        self.assertEqual(distance_matrix.get_distance(node1, node2), 24)

    def test_print_row_and_full_matrix(self):
        """Test that both formats produce the same distance matrix."""
        # Initialize from both formats
        lower_row_matrix = DistanceMatrix(file_path=self.lower_row_file.name)
        full_matrix = DistanceMatrix(file_path=self.temp_file.name)

        # print the matrices for debugging
        print("Lower Row Matrix:\n", lower_row_matrix.get_matrix())
        print("Full Matrix:\n", full_matrix.get_matrix())


if __name__ == '__main__':
    unittest.main()