from dataclasses import dataclass

from datastructure.metrics import BaseMetric


class Distance(BaseMetric['Distance']):
    """A strong type for distance values."""

    pass

@dataclass
class DistanceSegment:
    """
    DistanceSegment tracks statistics about route distance resulting from visiting clients in a specified order.

    Attributes:
        _distance: Total distance of a segment.
    """
    _distance: Distance = Distance(0)

    @property
    def distance(self) -> Distance:
        """Get the distance of the segment."""
        return self._distance

    @staticmethod
    def merge(edge_distance: Distance,
              first: 'DistanceSegment',
              second: 'DistanceSegment') -> 'DistanceSegment':
        """
        Merge two distance segments with an edge between them.

        Args:
            edge_distance: Distance of the edge connecting segments
            first: First distance segment
            second: Second distance segment

        Returns:
            A new merged distance segment
        """
        return DistanceSegment(
            _distance=Distance(first.distance + second.distance + edge_distance),
        )
