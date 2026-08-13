"""Student file: implement Random-Restart Hill Climbing here."""

from experiment import run_experiment
import numpy as np

# For a quick test, choose one benchmark and one seed, then run this file.
BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = [0]  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

# Add your chosen hyperparameters here.
PARAMETERS = {
    "sigma": 0.05,            # step size scale relative to domain range
    "patience": 200,          # steps without improvement before triggering a restart
    # "neighbors_per_step": 1,  # number of local neighbor samples evaluated per iteration
}


def hill_climbing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    sigma = parameters.get("sigma", 0.05)
    patience = parameters.get("patience", 200)
    
    lb = np.full(dimension, lower_bound, dtype=np.float64)
    ub = np.full(dimension, upper_bound, dtype=np.float64)
    domain_range = ub - lb

    evaluations = 0

    # helper function for uniform random candidate initialization inside bounds
    def random_point():
        return lb + rng.uniform(0.0, 1.0, size=dimension) * domain_range

    # global best tracker across all restarts
    best_x = None
    best_val = np.inf

    # main optimization loop bounded strictly by max_evaluations budget
    while evaluations < max_evaluations:
        # randomly restart
        current_x = random_point()
        current_val = objective(current_x)
        evaluations += 1

        if current_val < best_val:
            best_val = current_val
            best_x = np.copy(current_x)

        no_improvement_counter = 0

        # local exploitation
        while evaluations < max_evaluations and no_improvement_counter < patience:
            # generate a candidate neighbor via Gaussian mutation
            noise = rng.normal(0.0, sigma, size=dimension) * domain_range
            candidate_x = current_x + noise

            # boundary handling: clip candidate to bounds
            candidate_x = np.clip(candidate_x, lb, ub)

            # evaluate candidate
            candidate_val = objective(candidate_x)
            evaluations += 1

            # strict minimization check
            if candidate_val < current_val:
                current_x = candidate_x
                current_val = candidate_val
                no_improvement_counter = 0  # reset patience counter

                # update global best if overall champion improved
                if current_val < best_val:
                    best_val = current_val
                    best_x = np.copy(current_x)
            else:
                no_improvement_counter += 1  # increment stall counter

    return best_x, best_val

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

