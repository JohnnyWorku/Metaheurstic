"""Student file: implement an Evolution Strategy here."""

import numpy as np
from experiment import run_experiment

BENCHMARKS_TO_RUN = ["rosenbrock"]
SEEDS = [0]  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

# Algorithm Hyperparameters based on slides 24–25 & 29
PARAMETERS = {}


def evolution_strategy(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    mu = parameters.get("mu", 15)
    lambda_ = parameters.get("lambda_", 100)
    sigma_init = parameters.get("sigma_init", 0.5)
    
    # standard Schwefel learning rate for self-adaptation: tau ~ 1 / sqrt(2 * dimension)
    tau = parameters.get("tau")
    if tau is None:
        tau = 1.0 / np.sqrt(2.0 * dimension)

    # initialize mu parents uniformly within [lower_bound, upper_bound]
    parents_x = rng.uniform(lower_bound, upper_bound, size=(mu, dimension))
    parents_sigma = np.full((mu, 1), sigma_init)
    
    # evaluate initial parents
    parents_fitness = np.array([objective(x) for x in parents_x])
    evaluations_used = mu

    # track best overall solution
    best_idx = np.argmin(parents_fitness)
    best_x = parents_x[best_idx].copy()
    best_fitness = parents_fitness[best_idx]

    while evaluations_used < max_evaluations:
        # determine current generation's offspring count (respecting max_evaluations budget)
        current_lambda = min(lambda_, max_evaluations - evaluations_used)
        if current_lambda <= 0:
            break

        offspring_x = np.zeros((current_lambda, dimension))
        offspring_sigma = np.zeros((current_lambda, 1))
        offspring_fitness = np.zeros(current_lambda)

        for i in range(current_lambda):
            # Uniformly select a parent from the mu parents
            parent_idx = rng.integers(0, mu)
            p_x = parents_x[parent_idx]
            p_sigma = parents_sigma[parent_idx]

            # log-normal mutation for step size: sigma' = sigma * exp(tau * N(0,1))
            sigma_prime = p_sigma * np.exp(tau * rng.normal(0, 1))
            sigma_prime = np.maximum(sigma_prime, 1e-8)  # Prevent numerical underflow

            # gaussian mutation for object variables (Slide 29)
            x_prime = p_x + sigma_prime * rng.normal(0, 1, size=dimension)

            # enforce domain boundaries
            x_prime = np.clip(x_prime, lower_bound, upper_bound)

            offspring_x[i] = x_prime
            offspring_sigma[i] = sigma_prime

 
        for i in range(current_lambda):
            offspring_fitness[i] = objective(offspring_x[i])
            evaluations_used += 1

            # Update best seen solution
            if offspring_fitness[i] < best_fitness:
                best_fitness = offspring_fitness[i]
                best_x = offspring_x[i].copy()

        # truncation selection: pick the best mu individuals strictly from the offspring pool
        sorted_indices = np.argsort(offspring_fitness)
        selected_indices = sorted_indices[:min(mu, current_lambda)]

        parents_x = offspring_x[selected_indices]
        parents_sigma = offspring_sigma[selected_indices]
        parents_fitness = offspring_fitness[selected_indices]

    return best_x, best_fitness


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            evolution_strategy,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()
