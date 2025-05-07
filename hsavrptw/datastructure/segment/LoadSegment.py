class LoadSegment:
    def __init__(self, load=0, excess=0, capacity=float('inf')):
        self._load = load
        self._excess = excess
        self._capacity = capacity

    @property
    def load(self):
        return self._load

    @property
    def excess(self):
        return self._excess

    @property
    def capacity(self):
        return self._capacity

    @staticmethod
    def merge(first, second):
        total_load = first.load() + second.load()
        excess = max(0, total_load - first.capacity())
        result = LoadSegment(total_load, excess, first.capacity())
        return result