"""Student file: implement Simulated Annealing here."""

from experiment import run_experiment

BENCHMARKS_TO_RUN = ["rosenbrock"]
SEEDS = [0]  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000
PARAMETERS = {}  # Add your chosen hyperparameters here.


def simulated_annealing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    # TODO(student): implement and justify all algorithm design choices.
    raise NotImplementedError("Implement Simulated Annealing")


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            simulated_annealing,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()

