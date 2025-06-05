from dataclasses import dataclass
from statistics import fmean
from warnings import warn

import numpy as np

from datastructure import VRPProblem, VRPSolution
from datastructure.matrices.distance_matrix import DistanceMatrix
from datastructure.matrices.duration_matrix import DurationMatrix


class PenaltyBoundWarning(UserWarning):
    """
    Raised when a penalty parameter has reached its maximum value. This means
    software struggles to find a feasible solution for the instance that's being
    solved, either because the instance has no feasible solution, or it is just
    very hard to find one.
    """

@dataclass
class PenaltyParams:
    repair_booster: int = 12
    solutions_between_updates: int = 50
    penalty_increase: float = 1.34
    penalty_decrease: float = 0.32
    target_feasible: float = 0.43


class PenaltyManager:
    """
    Manages the penalty for infeasible solutions.
    Adjusts the penalty weights based on the feasibility of solutions found during the search.
    Transfer the new penalty weights to the cost evaluator to calculate the new solution costs.
    Parameters
    ----------
    params
        Parameters for the penalty manager.
    """
    MIN_PENALTY: float = 0.1
    MAX_PENALTY: float = 100_000.0
    FEAS_TOL: float = 0.05

    def __init__(self, initial_penalties: tuple[float, float], params: PenaltyParams):
        self._params = params
        self._penalties = np.clip(
            initial_penalties,
            self.MIN_PENALTY,
            self.MAX_PENALTY
        )

        self._feas_lists: list[list[bool]] = [
            [] for _ in range(len(self._penalties))
        ]

    @property
    def penalties(self) -> tuple[float, float]:
        return (
            float(self._penalties[0]), # excess load
            float(self._penalties[1]) # time warp
        )

    @property
    def boosted_penalties(self) -> tuple[float, float]:
        """
        Returns the boosted penalties, which are used for repairing infeasible solutions.
        """
        return (
            float(self._penalties[0]) * self._params.repair_booster,
            float(self._penalties[1]) * self._params.repair_booster
        )

    @classmethod
    def init_from(cls, problem: VRPProblem, params: PenaltyParams = PenaltyParams()) -> "PenaltyManager":
        """
        Initializes the penalty manager with the given problem data and parameters.
        """
        distances = DistanceMatrix(problem.nodes).get_matrix()
        durations = DurationMatrix(problem.nodes).get_matrix() if problem.type == "VRPTW" else None

        edge_costs = distances + durations if problem.type == "VRPTW" else distances
        avg_cost = edge_costs.mean()
        avg_duration = np.minimum.reduce(durations).mean() if durations is not None else 0
        avg_load = np.mean([node.demand for node in problem.nodes]) if len(problem.nodes) > 1 else 0

        # Initial penalty parameters are meant to weigh an average increase
        # in the relevant value by the same amount as the average edge cost.
        init_load = avg_cost / np.maximum(avg_load, 1)
        init_tw = avg_cost / max(avg_duration, 1)

        return cls((init_load, init_tw), params)


    def _compute(self, penalty: float, feas_percentage: float) -> float:
        # Computes and returns the new penalty value, given the current value
        # and the percentage of feasible solutions since the last update.
        diff = self._params.target_feasible - feas_percentage

        if abs(diff) < self.FEAS_TOL:
            return penalty

        if diff > 0:
            new_penalty = self._params.penalty_increase * penalty
        else:
            new_penalty = self._params.penalty_decrease * penalty

        if new_penalty >= self.MAX_PENALTY:
            msg = """
            A penalty parameter has reached its maximum value. This means software
            struggles to find a feasible solution for the instance that's being
            solved, either because the instance has no feasible solution, or it
            is very hard to find one. Check the instance carefully to determine
            if a feasible solution exists.
            """
            warn(msg, PenaltyBoundWarning)

        return np.clip(new_penalty, self.MIN_PENALTY, self.MAX_PENALTY)

    def _register(self, feas_list: list[bool], penalty: float, is_feas: bool):
        feas_list.append(is_feas)

        if len(feas_list) != self._params.solutions_between_updates:
            return penalty

        avg = fmean(feas_list)
        feas_list.clear()
        return self._compute(penalty, avg)

    def register(self, sol: VRPSolution):
        """
        Registers the feasibility dimensions of the given solution.
        """
        is_feasible = [
            sol.get_solution_penalties()[0] == 0,  # excess load
            sol.get_solution_penalties()[1] == 0,  # time warp
        ]

        for idx, is_feas in enumerate(is_feasible):
            feas_list = self._feas_lists[idx]
            penalty = self._penalties[idx]
            self._penalties[idx] = self._register(feas_list, penalty, is_feas)


class PenaltyWeights:
    _instances = {}

    def __new__(cls, load: float, time_warp: float):
        """
        Create a singleton instance of PenaltyWeights for a given load and time warp.
        """
        # Create a unique key based on load and time_warp
        instance_key = (load, time_warp)

        # Return existing instance if available
        if instance_key in cls._instances:
            return cls._instances[instance_key]

        # Create new instance
        instance = super().__new__(cls)
        cls._instances[instance_key] = instance
        return instance

    def __init__(self, load: float, time_warp: float):
        """
        Create an instance with penalty data.
        """
        if hasattr(self, 'initialized'):
            # Skip initialization if already initialized
            return
        self._load_penalty = load
        self._tw_penalty = time_warp

    @property
    def load_penalty(self) -> float:
        return self._load_penalty

    @property
    def time_window_penalty(self) -> float:
        return self._tw_penalty
