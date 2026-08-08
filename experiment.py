"""Small evaluation, logging, and plotting helpers shared by all algorithms."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from benchmarks import Benchmark, Vector, get_benchmark

PILOT_SEEDS = [0, 1, 2, 3, 4]
FINAL_SEEDS = list(range(100, 120))
EVALUATION_BUDGET = 20_000
LOG_EVERY = 100


class EvaluationBudgetReached(RuntimeError):
    """Raised if an algorithm attempts to exceed its evaluation budget."""


class Objective:
    """Count calls to a benchmark and remember the best solution seen."""

    def __init__(self, benchmark: Benchmark, budget: int, log_every: int = LOG_EVERY):
        self.benchmark = benchmark
        self.budget = budget
        self.log_every = log_every
        self.evaluations = 0
        self.best_candidate: Vector | None = None
        self.best_value = float("inf")
        self.history: list[tuple[int, float]] = []

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    def __call__(self, candidate: Vector) -> float:
        if self.remaining == 0:
            raise EvaluationBudgetReached(
                f"The objective evaluation budget ({self.budget}) is exhausted."
            )

        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != (self.benchmark.dimension,):
            raise ValueError(
                f"Candidate must have shape ({self.benchmark.dimension},)."
            )
        if np.any(candidate < self.benchmark.lower_bound) or np.any(
            candidate > self.benchmark.upper_bound
        ):
            raise ValueError("Candidate is outside the benchmark domain.")

        value = self.benchmark.function(candidate)
        self.evaluations += 1
        if value < self.best_value:
            self.best_value = value
            self.best_candidate = candidate.copy()
        if self.evaluations % self.log_every == 0 or self.evaluations == self.budget:
            self.history.append((self.evaluations, self.best_value))
        return value


Algorithm = Callable[..., Any]


@dataclass
class RunResult:
    algorithm: str
    benchmark: str
    seed: int
    best_candidate: Vector
    best_value: float
    evaluations: int
    history: list[tuple[int, float]]


def run_once(
    algorithm: Algorithm,
    benchmark_name: str,
    seed: int,
    parameters: dict[str, Any] | None = None,
    max_evaluations: int = EVALUATION_BUDGET,
) -> RunResult:
    """Run one algorithm once. The wrapper enforces the objective-call budget."""
    benchmark = get_benchmark(benchmark_name)
    objective = Objective(benchmark, max_evaluations)
    rng = np.random.default_rng(seed)

    algorithm(
        objective=objective,
        lower_bound=benchmark.lower_bound,
        upper_bound=benchmark.upper_bound,
        dimension=benchmark.dimension,
        rng=rng,
        max_evaluations=max_evaluations,
        **(parameters or {}),
    )

    if objective.evaluations != max_evaluations:
        raise ValueError(
            f"Algorithm used {objective.evaluations} evaluations; "
            f"it must use exactly {max_evaluations}."
        )
    assert objective.best_candidate is not None
    return RunResult(
        algorithm=algorithm.__name__,
        benchmark=benchmark.name,
        seed=seed,
        best_candidate=objective.best_candidate,
        best_value=objective.best_value,
        evaluations=objective.evaluations,
        history=objective.history,
    )


def run_experiment(
    algorithm: Algorithm,
    benchmark_name: str,
    seeds: list[int],
    parameters: dict[str, Any] | None = None,
    output_directory: str = "results",
    max_evaluations: int = EVALUATION_BUDGET,
) -> list[RunResult]:
    """Run several seeds, save CSV files, and create two simple plots."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    results = [
        run_once(algorithm, benchmark_name, seed, parameters, max_evaluations)
        for seed in seeds
    ]

    stem = f"{algorithm.__name__}_{benchmark_name}"
    _save_results(results, output / f"{stem}_results.csv")
    _save_history(results, output / f"{stem}_history.csv")
    plot_convergence(results, output / f"{stem}_convergence.png")
    plot_final_values(results, output / f"{stem}_final_values.png")
    return results


def _save_results(results: list[RunResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["algorithm", "benchmark", "seed", "evaluations", "best_value", "best_candidate"]
        )
        for result in results:
            writer.writerow(
                [
                    result.algorithm,
                    result.benchmark,
                    result.seed,
                    result.evaluations,
                    result.best_value,
                    result.best_candidate.tolist(),
                ]
            )


def _save_history(results: list[RunResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["algorithm", "benchmark", "seed", "evaluations", "best_value"])
        for result in results:
            for evaluations, best_value in result.history:
                writer.writerow(
                    [result.algorithm, result.benchmark, result.seed, evaluations, best_value]
                )


def plot_convergence(results: list[RunResult], path: Path) -> None:
    """Plot best-so-far curves; multiple seeds are summarized by median and IQR."""
    evaluations = np.asarray([point[0] for point in results[0].history])
    values = np.asarray([[point[1] for point in result.history] for result in results])
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(evaluations, median, label="median best-so-far")
    plt.fill_between(evaluations, q1, q3, alpha=0.25, label="IQR")
    plt.xlabel("Objective evaluations")
    plt.ylabel("Objective value (lower is better)")
    plt.title(f"{results[0].algorithm} on {results[0].benchmark}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_final_values(results: list[RunResult], path: Path) -> None:
    """Plot the final best values across seeds."""
    plt.figure(figsize=(5, 4))
    plt.boxplot([result.best_value for result in results], tick_labels=[results[0].benchmark])
    plt.ylabel("Final objective value (lower is better)")
    plt.title(results[0].algorithm)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def print_result(result: RunResult) -> None:
    print(f"Benchmark: {result.benchmark}")
    print(f"Seed: {result.seed}")
    print(f"Evaluations: {result.evaluations}")
    print(f"Best candidate: {result.best_candidate}")
    print(f"Best objective: {result.best_value}")
