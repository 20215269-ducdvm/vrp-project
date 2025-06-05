from numpy.testing import assert_equal

from datastructure.penalty_manager import PenaltyParams, PenaltyManager


def test_repair_booster():
    """
    Tests that the booster evaluator returns a cost evaluator object that
    penalises constraint violations much more severely.
    """
    params = PenaltyParams(5, 1, 1, 1, 1)
    pm = PenaltyManager((1, 1), params)

    penalties = pm.penalties

    assert_equal(penalties[0], 1)
    assert_equal(penalties[1], 1)  # 1 unit above cap

    # With the booster, the penalty values are multiplied by the
    # repairBooster term (x5 in this case).
    booster = pm.boosted_penalties
    assert_equal(booster[0], 5)
    assert_equal(booster[1], 5)
