from dataclasses import dataclass, field


@dataclass
class Node:
    node_id: int
    x_coordinate: float
    y_coordinate: float
    demand: int
    is_depot: bool
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self):
        self._initialized = True

    def __setattr__(self, name, value):
        if getattr(self, '_initialized', False) and name != '_initialized' and name in self.__annotations__:
            raise AttributeError(f"Cannot modify immutable attribute '{name}'")
        super().__setattr__(name, value)

    def __repr__(self):
        return str(self.node_id)

    def __lt__(self, other):
        return self.node_id > other.node_id

    def __eq__(self, other):
        return self.node_id == other.node_id

    def __hash__(self):
        return self.node_id


@dataclass
class NodeWithTW(Node):
    time_window: tuple[int, int] = None
    service_time: int = 0
    arrival_time: int = 0
    waiting_time: int = 0

    def __hash__(self):
        return super().__hash__()

    def __setattr__(self, name, value):
        if name in ['arrival_time', 'waiting_time']:
            # Allow changes to these attributes
            super(Node, self).__setattr__(name, value)
        else:
            # Use parent's protection for other attributes
            super().__setattr__(name, value)