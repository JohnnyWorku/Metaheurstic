import numpy as np
import pytest

from benchmarks import get_benchmark
from experiment import EvaluationBudgetReached, Objective, run_once
from random_search import random_search


def test_objective_counts_calls_and_stops_at_budget():
    objective = Objective(get_benchmark("sphere"), budget=3, log_every=1)
    objective(np.ones(10))
    objective(np.zeros(10))
    objective(np.full(10, 2.0))
    assert objective.evaluations == 3
    assert objective.best_value == 0.0
    assert len(objective.history) == 3
    with pytest.raises(EvaluationBudgetReached):
        objective(np.zeros(10))


def test_objective_rejects_wrong_shape_and_out_of_domain_without_counting():
    objective = Objective(get_benchmark("sphere"), budget=3)
    with pytest.raises(ValueError):
        objective(np.zeros(2))
    with pytest.raises(ValueError):
        objective(np.full(10, 100.0))
    assert objective.evaluations == 0


def test_random_search_is_reproducible_and_uses_exact_budget():
    first = run_once(random_search, "rastrigin", seed=7, max_evaluations=25)
    second = run_once(random_search, "rastrigin", seed=7, max_evaluations=25)
    assert first.evaluations == 25
    assert first.best_value == second.best_value
    np.testing.assert_array_equal(first.best_candidate, second.best_candidate)


def test_runner_rejects_an_algorithm_that_stops_early():
    def early_stop(**arguments):
        candidate = arguments["rng"].uniform(
            arguments["lower_bound"],
            arguments["upper_bound"],
            arguments["dimension"],
        )
        arguments["objective"](candidate)

    with pytest.raises(ValueError, match="exactly"):
        run_once(early_stop, "sphere", seed=0, max_evaluations=2)

