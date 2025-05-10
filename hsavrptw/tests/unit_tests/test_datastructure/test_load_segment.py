import unittest

from datastructure.segment.LoadSegment import Load


class TestLoad(unittest.TestCase):
    def test_load_arithmetic(self):
        l1 = Load(10)
        l2 = Load(5)

        self.assertEqual(Load(15), l1 + l2)
        self.assertEqual(Load(15), 10 + l2)
        self.assertEqual(Load(5), l1 - l2)
        self.assertEqual(Load(5), 10 - l2)
        self.assertEqual(Load(-10), -l1)
        self.assertEqual(Load(10), +l1)

    def test_load_comparison(self):
        l1 = Load(10)
        l2 = Load(5)

        self.assertTrue(l1 > l2)
        self.assertTrue(l1 >= l2)
        self.assertFalse(l1 < l2)
        self.assertFalse(l1 <= l2)
        self.assertTrue(l2 < l1)
        self.assertTrue(l2 <= l1)

class TestLoadSegment(unittest.TestCase):
    def test_merge_basic(self):
        # Create two simple segments
        first = Load(100)
        second = Load(150)

        # Merge the segments
        merged = first + second

        # Test the merged segment
        self.assertEqual(Load(250), merged)  # 100 + 150 + 20