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
        _demand: Total load of a segment.
    """
    _demand: Load = Load(0)

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
            _demand=Load(first.load + second.load),
        )

    @property
    def load(self):
        return self._demand

    @classmethod
    def from_client(cls, client) -> 'LoadSegment':
        """
        Create a LoadSegment from a client.

        Args:
            client: The client to create the LoadSegment from.

        Returns:
            A new LoadSegment instance.
        """
        return cls(
            _demand=Load(client.demand),
        )