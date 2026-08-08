"""Student file: implement Random-Restart Hill Climbing here."""

from experiment import run_experiment

# For a quick test, choose one benchmark and one seed, then run this file.
BENCHMARKS_TO_RUN = ["rosenbrock"]
SEEDS = [0]  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

# Add your chosen hyperparameters here.
PARAMETERS = {}


def hill_climbing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    # TODO(student): implement and justify all algorithm design choices.
    raise NotImplementedError("Implement Random-Restart Hill Climbing")


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            hill_climbing,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()

