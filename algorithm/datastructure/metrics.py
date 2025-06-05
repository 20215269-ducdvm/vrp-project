from typing import TypeVar, Generic, cast

T = TypeVar('T', bound='BaseMetric')


class BaseMetric(int, Generic[T]):
    """Base class for metric values like Duration and Load."""

    def __new__(cls, value: int) -> T:
        instance = super(BaseMetric, cls).__new__(cls, value)
        return cast(T, instance)

    def __add__(self, other) -> T:
        return self.__class__(super().__add__(other))

    def __radd__(self, other) -> T:
        return self.__class__(super().__radd__(other))

    def __sub__(self, other) -> T:
        return self.__class__(super().__sub__(other))

    def __rsub__(self, other) -> T:
        return self.__class__(super().__rsub__(other))

    def __neg__(self) -> T:
        return self.__class__(super().__neg__())

    def __pos__(self) -> T:
        return self.__class__(super().__pos__())

    def __mul__(self, other) -> T:
        return self.__class__(super().__mul__(other))

    def __rmul__(self, other) -> T:
        return self.__class__(super().__rmul__(other))

    # Comparison operators don't need to return the specialized type
    def __lt__(self, other) -> bool:
        return super().__lt__(other)

    def __le__(self, other) -> bool:
        return super().__le__(other)

    def __gt__(self, other) -> bool:
        return super().__gt__(other)

    def __ge__(self, other) -> bool:
        return super().__ge__(other)