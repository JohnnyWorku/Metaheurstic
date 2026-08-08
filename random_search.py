"""Working example showing how to use the small experiment engine."""

from experiment import run_experiment

# Change these values and run: python random_search.py
BENCHMARKS_TO_RUN = ["rosenbrock"]
SEEDS = [0]
MAX_EVALUATIONS = 20_000
PARAMETERS = {}


def random_search(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
):
    while objective.remaining > 0:
        candidate = rng.uniform(lower_bound, upper_bound, size=dimension)
        objective(candidate)


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            algorithm=random_search,
            benchmark_name=benchmark_name,
            seeds=SEEDS,
            parameters=PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(
            f"{benchmark_name}: best={results[0].best_value:.6g}; "
            "CSV logs and plots saved in results/"
        )


if __name__ == "__main__":
    main()

