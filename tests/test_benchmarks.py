import numpy as np
import pytest

from benchmarks import BENCHMARKS, get_benchmark, rastrigin, rosenbrock, sphere


@pytest.mark.parametrize("name", ["sphere", "rastrigin", "rosenbrock"])
def test_benchmark_optimum_and_dimension(name):
    benchmark = get_benchmark(name)
    assert benchmark.dimension == 10
    assert benchmark.function(benchmark.optimum) == pytest.approx(0.0, abs=1e-12)


def test_benchmark_formulas():
    assert sphere(np.array([1.0, 2.0])) == pytest.approx(5.0)
    assert rastrigin(np.ones(3)) == pytest.approx(3.0)
    assert rosenbrock(np.array([0.0, 0.0])) == pytest.approx(1.0)


def test_domains():
    assert (BENCHMARKS["sphere"].lower_bound, BENCHMARKS["sphere"].upper_bound) == (
        -5.12,
        5.12,
    )
    assert (
        BENCHMARKS["rastrigin"].lower_bound,
        BENCHMARKS["rastrigin"].upper_bound,
    ) == (-5.12, 5.12)
    assert (
        BENCHMARKS["rosenbrock"].lower_bound,
        BENCHMARKS["rosenbrock"].upper_bound,
    ) == (-5.0, 10.0)

