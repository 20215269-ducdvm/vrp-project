from dataclasses import dataclass
from typing import Optional
import sys

from datastructure.metrics import BaseMetric


class Duration(BaseMetric['Duration']):
    """A strong type for duration values."""

    pass
    
@dataclass
class DurationSegment:
    """
    DurationSegment tracks statistics about route duration and time warp
    resulting from visiting clients in a specified order.

    Attributes:
        _duration: Total duration of a segment, including waiting time and service time.
        Start from the moment where first client is served, ends at the moment where the vehicle finishes serving the last client.
        _time_warp: Total time warp on the route segment. Time warp occurs when the vehicle arrives at a client after its time window has closed. Vehicle then travels back to the moment when the time window closes.
        _tw_early: Earliest visit moment of the first client.
        _tw_late: Latest visit moment of the first client.
        _release_time: Earliest moment to start the route segment.
    """
    _duration: Duration = Duration(0)
    _time_warp: Duration = Duration(0)
    _tw_early: Duration = Duration(0)
    _tw_late: Duration = Duration(0)
    _release_time: Duration = Duration(0)

    @staticmethod
    def merge(edge_duration: Duration,
              first: 'DurationSegment',
              second: 'DurationSegment') -> 'DurationSegment':
        """
        Merge two duration segments with an edge between them.

        Args:
            edge_duration: Duration of the edge connecting segments
            first: First duration segment
            second: Second duration segment

        Returns:
            A new merged duration segment
        """
        # Skip calculation if time windows are disabled
        if hasattr(sys, "VRP_NO_TIME_WINDOWS"):
            return DurationSegment()

        # atSecond is the time it takes to travel to the second segment from the moment the vehicle start serving at
        # the first segment
        at_second = Duration(first._duration - first._time_warp + edge_duration)

        # diff_tw is the amount of time the vehicle arrives late at the second segment
        diff_tw = Duration(max(0, first._tw_early + at_second - second._tw_late))

        # diff_wait is the amount of time the vehicle has to inevitably wait at the second segment. assume that even
        # if the vehicle starts the first segment at the latest possible moment (w/o time warp), it will still arrive
        # at the second segment too early.
        diff_wait = Duration(max(Duration(0), Duration(second._tw_early - at_second - first._tw_late)))

        # New twLate for the second segment
        max_duration = Duration(sys.maxsize)
        second_late = Duration(second._tw_late - at_second) if at_second > Duration(second._tw_late - max_duration) \
            else Duration(second._tw_late)

        return DurationSegment(
            _duration=Duration(first._duration + second._duration + edge_duration + diff_wait),
            _time_warp=Duration(first._time_warp + second._time_warp + diff_tw),
            _tw_early=Duration(max(Duration(second._tw_early - at_second), first._tw_early) - diff_wait),
            _tw_late=Duration(min(second_late, first._tw_late) + diff_tw),
            _release_time=Duration(max(Duration(first._release_time), Duration(second._release_time)))
        )

    def time_warp(self, max_duration: Optional[Duration] = None) -> Duration:
        """
        Returns the time warp on this route segment. Additionally, any time warp
        incurred by violating the maximum duration argument is also counted.

        Args:
            max_duration: Maximum allowed duration, if provided. If the segment's duration
                exceeds this value, any excess duration is counted as time warp.
                Default is unlimited.

        Returns:
            Total time warp on this route segment.
        """
        if max_duration is None:
            max_duration = Duration(sys.maxsize)

        excess_duration = Duration(max(Duration(0), Duration((self._duration - self._time_warp) - max_duration)))
        release_time_violation = Duration(max(Duration(0), Duration(self._release_time - self._tw_late)))

        return Duration(self._time_warp + release_time_violation + excess_duration)


    @property
    def duration(self) -> Duration:
        """The total duration of this route segment."""
        return self._duration

    @property
    def tw_early(self) -> Duration:
        """Earliest start time for this route segment that results in minimum route segment duration."""
        return self._tw_early

    @property
    def tw_late(self) -> Duration:
        """Latest start time for this route segment that results in minimum route segment duration."""
        return self._tw_late

    @property
    def release_time(self) -> Duration:
        """Earliest possible release time of the clients in this route segment."""
        return self._release_time

    # Factory methods to create instances from different data sources
    @classmethod
    def from_client(cls, client):
        """Construct from attributes of the given client."""
        return cls(
            _duration=Duration(client.duration),
            _time_warp=Duration(0),
            _tw_early=Duration(client.tw_early),
            _tw_late=Duration(client.tw_late),
            _release_time=Duration(client.release_time)
        )

    @classmethod
    def from_depot(cls, depot, service_duration: Duration):
        """Construct from attributes of the given depot and depot service duration."""
        return cls(
            _duration=service_duration,
            _time_warp=Duration(0),
            _tw_early=Duration(depot.tw_early),
            _tw_late=Duration(depot.tw_late),
            _release_time=Duration(depot.release_time)
        )

