import numpy as np
import pytest

from benchmarks import get_benchmark
from evolution_strategy import evolution_strategy
from experiment import Objective
from genetic_algorithm import genetic_algorithm
from hill_climbing import hill_climbing
from simulated_annealing import simulated_annealing


@pytest.mark.parametrize(
    "algorithm",
    [hill_climbing, simulated_annealing, evolution_strategy, genetic_algorithm],
)
def test_required_algorithms_are_left_for_students(algorithm):
    benchmark = get_benchmark("sphere")
    with pytest.raises(NotImplementedError):
        algorithm(
            objective=Objective(benchmark, 10),
            lower_bound=benchmark.lower_bound,
            upper_bound=benchmark.upper_bound,
            dimension=benchmark.dimension,
            rng=np.random.default_rng(0),
            max_evaluations=10,
        )
