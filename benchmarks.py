"""The three benchmark functions used in the assignment."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class Benchmark:
    name: str
    function: Callable[[Vector], float]
    lower_bound: float
    upper_bound: float
    dimension: int
    optimum: Vector


def sphere(x: Vector) -> float:
    """Sphere: global minimum 0 at x = (0, ..., 0)."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def rastrigin(x: Vector) -> float:
    """Rastrigin: global minimum 0 at x = (0, ..., 0)."""
    x = np.asarray(x, dtype=float)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x: Vector) -> float:
    """Rosenbrock: global minimum 0 at x = (1, ..., 1)."""
    x = np.asarray(x, dtype=float)
    return float(
        np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    )


BENCHMARKS = {
    "sphere": Benchmark("sphere", sphere, -5.12, 5.12, 10, np.zeros(10)),
    "rastrigin": Benchmark("rastrigin", rastrigin, -5.12, 5.12, 10, np.zeros(10)),
    "rosenbrock": Benchmark("rosenbrock", rosenbrock, -5.0, 10.0, 10, np.ones(10)),
}


def get_benchmark(name: str) -> Benchmark:
    """Return a benchmark, its bounds, and its dimension by name."""
    try:
        return BENCHMARKS[name.lower()]
    except KeyError as error:
        raise ValueError(f"Choose one of: {', '.join(BENCHMARKS)}") from error


if __name__ == "__main__":
    for benchmark in BENCHMARKS.values():
        value = benchmark.function(benchmark.optimum)
        print(
            f"{benchmark.name}: dimension={benchmark.dimension}, "
            f"domain=[{benchmark.lower_bound}, {benchmark.upper_bound}], "
            f"f(optimum)={value}"
        )

