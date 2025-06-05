from dataclasses import dataclass

from datastructure.metrics import BaseMetric


class Load(BaseMetric['Load']):
    """A strong type for load values."""
    pass

@dataclass
class LoadSegment:
    """
    LoadSegment tracks statistics about route load resulting from visiting clients in a specified order.

    Attributes:
        _total_load: Total load of a segment.
    """
    _total_load: Load = Load(0)

    @staticmethod
    def merge(first: 'LoadSegment',
              second: 'LoadSegment') -> 'LoadSegment':
        """
        Merge two load segments with an edge between them.

        Args:
            edge_load: Load of the edge connecting segments
            first: First load segment
            second: Second load segment

        Returns:
            A new merged load segment
        """

        return LoadSegment(
            _total_load=Load(first.load + second.load),
        )

    @property
    def load(self):
        return self._total_load

    def excess_load(self, max_capacity) -> Load:
        return self._total_load - max_capacity if self._total_load > max_capacity else Load(0)