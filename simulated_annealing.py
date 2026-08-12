"""Student file: implement Simulated Annealing here."""

import numpy as np
from experiment import run_experiment

BENCHMARKS_TO_RUN = ["rosenbrock"]
SEEDS = [0]  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000
# Recommended Hyperparameters
PARAMETERS = {
    "initial_temperature": 1.0,
    "min_temperature": 1e-4,
    "alpha": 0.99954,  # smooth cooling decay across 20,000 evaluations
    "sigma": 0.02,     # mutation step size (as fraction of domain range)
}


def simulated_annealing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    # extract hyperparameters with sensible defaults
    T_0 = parameters.get("initial_temperature", 1.0)
    T_min = parameters.get("min_temperature", 1e-4)
    alpha = parameters.get("alpha", 0.99954)
    sigma = parameters.get("sigma", 0.02)

    # precompute domain vector properties for fast scaling
    lb = np.asarray(lower_bound)
    ub = np.asarray(upper_bound)
    domain_range = ub - lb

    # 1. initialize start position uniformly at random inside domain bounds
    current_x = lb + rng.uniform(0.0, 1.0, size=dimension) * domain_range
    current_val = objective(current_x)
    evaluations = 1

    # keep track of global best solution (champion)
    best_x = np.copy(current_x)
    best_val = current_val

    # set initial temperature
    T = T_0

    # 2. main optimization loop
    while evaluations < max_evaluations:
        # generate a candidate neighbor via Gaussian mutation
        noise = rng.normal(0.0, sigma, size=dimension) * domain_range
        candidate_x = current_x + noise

        # boundary handling: clip candidate to feasible domain
        candidate_x = np.clip(candidate_x, lb, ub)

        # evaluate candidate solution
        candidate_val = objective(candidate_x)
        evaluations += 1

        # calculate change in objective value (energy delta)
        delta_E = candidate_val - current_val

        # metropolis acceptance criterion (minimization)
        if delta_E <= 0:
            # strictly better or equal candidate -> Always accept
            accept = True
        else:
            # worse candidate -> cccept probabilistically based on temperature
            # clamp exponent to prevent underflow overflow warnings
            exponent = -delta_E / max(T, 1e-12)
            p_accept = np.exp(np.clip(exponent, -500, 0))
            accept = rng.uniform(0.0, 1.0) < p_accept

        # perform state transition if accepted
        if accept:
            current_x = candidate_x
            current_val = candidate_val

            # update global champion memory if overall record is broken
            if current_val < best_val:
                best_val = current_val
                best_x = np.copy(current_x)

        # 3. cool down temperature exponentially
        T = max(T * alpha, T_min)

    return best_x, best_val


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
