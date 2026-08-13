"""Student file: implement a Genetic Algorithm here."""

import numpy as np
from experiment import run_experiment

BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
SEEDS = [0]  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

# Recommended Hyperparameters
PARAMETERS = {
    "pop_size": 50,           # number of individuals in the population
    "tournament_size": 3,     # number of candidates in tournament selection
    "crossover_prob": 0.8,    # probability of applying crossover between parents
    "mutation_prob": 0.2,     # probability of mutating a gene (vector component)
    "sigma": 0.05,            # mutation magnitude (fraction of domain range)
    "elitism": True,          # pass the overall best solution directly to next generation
}


def genetic_algorithm(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    **parameters,
):
    # hyperparameters
    pop_size = parameters.get("pop_size", 50)
    tournament_size = parameters.get("tournament_size", 3)
    crossover_prob = parameters.get("crossover_prob", 0.8)
    mutation_prob = parameters.get("mutation_prob", 0.2)
    sigma = parameters.get("sigma", 0.05)
    elitism = parameters.get("elitism", True)

    # convert domain bounds to NumPy arrays
    lb = np.asarray(lower_bound)
    ub = np.asarray(upper_bound)
    domain_range = ub - lb

    # 1. initialize Population Uniformly at Random
    population = lb + rng.uniform(0.0, 1.0, size=(pop_size, dimension)) * domain_range
    fitness = np.zeros(pop_size)
    evaluations = 0

    # evaluate initial population
    for i in range(pop_size):
        if evaluations < max_evaluations:
            fitness[i] = objective(population[i])
            evaluations += 1
        else:
            break

    # track Global Champion (Best overall solution)
    best_idx = np.argmin(fitness[:evaluations])
    best_x = np.copy(population[best_idx])
    best_val = fitness[best_idx]

    # helper Function: tournament selection
    def select_parent():
        # pick k random indices from population
        candidates = rng.choice(pop_size, size=tournament_size, replace=False)
        # winner is the candidate with the lowest objective value (minimization)
        winner = candidates[np.argmin(fitness[candidates])]
        return population[winner]

    # main generational loop
    while evaluations < max_evaluations:
        new_population = []
        
        # elitism: preserve the best solution from current generation
        if elitism:
            curr_best_idx = np.argmin(fitness)
            new_population.append(np.copy(population[curr_best_idx]))

        # produce remaining population members via Selection, Crossover, and Mutation
        while len(new_population) < pop_size:
            parent1 = select_parent()
            parent2 = select_parent()

            # crossover phase (arithmetic blend crossover)
            if rng.uniform(0.0, 1.0) < crossover_prob:
                gamma = rng.uniform(0.0, 1.0, size=dimension)
                offspring = gamma * parent1 + (1.0 - gamma) * parent2
            else:
                offspring = np.copy(parent1)

            # mutation phase (gaussian mutation)
            mutation_mask = rng.uniform(0.0, 1.0, size=dimension) < mutation_prob
            if np.any(mutation_mask):
                noise = rng.normal(0.0, sigma, size=dimension) * domain_range
                offspring += mutation_mask * noise   # multiply by mask (0 - false and 1 - true) to apply mutation only to selected genes

            # boundary handling: clip to feasible domain bounds
            offspring = np.clip(offspring, lb, ub)
            new_population.append(offspring)

        # truncate if excess individuals were generated
        population = np.array(new_population[:pop_size])

        # evaluate the new generation
        for i in range(pop_size):
            if evaluations < max_evaluations:
                # skip re-evaluation for preserved elite individual at index 0
                if elitism and i == 0:
                    curr_val = best_val
                else:
                    curr_val = objective(population[i])
                    evaluations += 1

                fitness[i] = curr_val

                # update Global Champion Record
                if curr_val < best_val:
                    best_val = curr_val
                    best_x = np.copy(population[i])
            else:
                break

    return best_x, best_val


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            genetic_algorithm,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()