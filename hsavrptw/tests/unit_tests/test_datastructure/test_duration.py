import unittest

from datastructure.segment.DurationSegment import DurationSegment, Duration


class TestDuration(unittest.TestCase):
    def test_duration_arithmetic(self):
        d1 = Duration(10)
        d2 = Duration(5)

        self.assertEqual(Duration(15), d1 + d2)
        self.assertEqual(Duration(15), 10 + d2)
        self.assertEqual(Duration(5), d1 - d2)
        self.assertEqual(Duration(5), 10 - d2)
        self.assertEqual(Duration(-10), -d1)
        self.assertEqual(Duration(10), +d1)
        self.assertEqual(Duration(20), d1 * 2)
        self.assertEqual(Duration(20), 2 * d1)

    def test_duration_comparison(self):
        d1 = Duration(10)
        d2 = Duration(5)

        self.assertTrue(d1 > d2)
        self.assertTrue(d1 >= d2)
        self.assertFalse(d1 < d2)
        self.assertFalse(d1 <= d2)
        self.assertTrue(d2 < d1)
        self.assertTrue(d2 <= d1)


class TestDurationSegment(unittest.TestCase):
    def test_merge_basic(self):
        # Create two simple segments
        first = DurationSegment(
            _duration=Duration(100),
            _time_warp=Duration(0),
            _tw_early=Duration(10),
            _tw_late=Duration(50),
            _release_time=Duration(5)
        )

        second = DurationSegment(
            _duration=Duration(150),
            _time_warp=Duration(0),
            _tw_early=Duration(120),
            _tw_late=Duration(200),
            _release_time=Duration(10)
        )

        edge_duration = Duration(20)

        # Merge the segments
        merged = DurationSegment.merge(edge_duration, first, second)

        # Test the merged segment
        self.assertEqual(Duration(270), merged._duration)  # 100 + 150 + 20
        self.assertEqual(Duration(0), merged._time_warp)
        self.assertEqual(Duration(10), merged._tw_early)
        self.assertEqual(Duration(50), merged._tw_late)
        self.assertEqual(Duration(10), merged._release_time)  # max(5, 10)

    def test_merge_with_time_warp(self):
        first = DurationSegment(
            _duration=Duration(100),
            _time_warp=Duration(0),
            _tw_early=Duration(10),
            _tw_late=Duration(50),
            _release_time=Duration(5)
        )

        # Second segment has an early time window that causes time warp
        second = DurationSegment(
            _duration=Duration(150),
            _time_warp=Duration(10),
            _tw_early=Duration(90),
            _tw_late=Duration(100),
            _release_time=Duration(10)
        )

        edge_duration = Duration(20)

        # Merge the segments
        merged = DurationSegment.merge(edge_duration, first, second)

        # Arriving at second at time 130 (earliest moment of service 10 + total duration time of first segment 100 + travel time to next segment 20)
        # This is after the late window (100), so time warp of 30 occurs
        self.assertEqual(Duration(270), merged._duration)
        self.assertEqual(Duration(40), merged._time_warp)  # 10 from second + 30 for arriving late

    def test_merge_with_waiting_time(self):
        first = DurationSegment(
            _duration=Duration(50),
            _time_warp=Duration(0),
            _tw_early=Duration(10),
            _tw_late=Duration(50),
            _release_time=Duration(5)
        )

        # Second segment has a late time window that causes waiting
        second = DurationSegment(
            _duration=Duration(150),
            _time_warp=Duration(0),
            _tw_early=Duration(200),
            _tw_late=Duration(250),
            _release_time=Duration(10)
        )

        edge_duration = Duration(20)

        # Merge the segments
        merged = DurationSegment.merge(edge_duration, first, second)

        # The latest possible arrival time for the second segment w/o time warp is 50 (first segment tw_late) + 50 (duration of segment 1) + 20 (travel time from seg1 to seg2) = 120
        # But the earliest time window is 200, so the vehicle has to wait 80
        # total duration = 50 (duration of segment 1) + 20 (travel time) + 80 (waiting time) + 150 (duration of segment 2) = 300
        self.assertEqual(Duration(300), merged._duration)
        self.assertEqual(Duration(0), merged._time_warp)

    def test_time_warp_property(self):
        segment = DurationSegment(
            _duration=Duration(200),
            _time_warp=Duration(20),
            _tw_early=Duration(10),
            _tw_late=Duration(50),
            _release_time=Duration(60)  # Release time violation: can only begin service at 60, but latest arrival time is 50
        )

        # Default (no max duration)
        self.assertEqual(Duration(30), segment.time_warp())  # 20 + 10 (release_time - tw_late)

        # With max duration constraint
        self.assertEqual(Duration(210), segment.time_warp(Duration(0)))  # 20 (own time warp) + (200 - 20 - 0) (excess duration) + 10 (release time violation)

    def test_properties(self):
        segment = DurationSegment(
            _duration=Duration(200),
            _time_warp=Duration(20),
            _tw_early=Duration(10),
            _tw_late=Duration(50),
            _release_time=Duration(30)
        )

        self.assertEqual(Duration(200), segment.duration)
        self.assertEqual(Duration(10), segment.tw_early)
        self.assertEqual(Duration(50), segment.tw_late)
        self.assertEqual(Duration(30), segment.release_time)


if __name__ == '__main__':
    unittest.main()