# Optimization Using Metaheuristics

This is a small, function-based starter repository. The infrastructure provides
the three test problems, fair objective-evaluation counting, CSV logging, and
plots. Implement the algorithms and make the algorithm-design choices.

## Install

Python 3.11 or newer is required.

```bash
python -m pip install -e '.[test]'
```

The only runtime dependencies are NumPy and Matplotlib. Pytest is included in
the optional test dependencies.

## Try a benchmark

```bash
python benchmarks.py
python random_search.py
```

`benchmarks.py` prints the domain, dimension, and optimum check for Sphere,
Rastrigin, and Rosenbrock. `random_search.py` is a working example of the engine;
it writes CSV logs and plots into `results/`.

To work on an algorithm, open its file:

- `hill_climbing.py`
- `simulated_annealing.py`
- `evolution_strategy.py`
- `genetic_algorithm.py`

Choose the benchmark at the top:

```python
BENCHMARKS_TO_RUN = ["rosenbrock"]
SEEDS = [0]
PARAMETERS = {}
```

Implement the function and run the file directly:

```bash
python hill_climbing.py
```

Use `SEEDS = [0, 1, 2, 3, 4]` for pilot experiments. After choosing and
freezing your design, use `SEEDS = range(100, 120)` for final experiments. To
run all problems, set:

```python
BENCHMARKS_TO_RUN = ["sphere", "rastrigin", "rosenbrock"]
```

## What the engine provides

Every algorithm receives these arguments:

- `objective`: call this once for each candidate that you evaluate;
- `lower_bound`, `upper_bound`, and `dimension`;
- `rng`: a reproducible `numpy.random.default_rng(seed)` generator;
- `max_evaluations`: the fixed objective-call limit;
- the contents of your own `PARAMETERS` dictionary.

Use `objective.remaining` to avoid exceeding the budget. Initialization counts.
Population members count separately. The runner rejects early stopping and the
objective raises an error before a call beyond the budget.

The engine does **not** provide iteration counts. An iteration has different
costs for a single-state algorithm and a population algorithm, so the common
fair limit is 20,000 objective evaluations. Your algorithm may maintain its own
iteration or generation counter if it needs one.

## What interns choose and implement

Implement random candidate creation, selection, reproduction, joining
or survivor replacement, boundary handling, and all algorithm-specific
hyperparameters. Examples of choices discussed in training include truncation,
tournament or roulette selection; Gaussian or bounded-uniform changes;
one-point, two-point, uniform, or line crossover; restart and cooling behavior;
and `(mu, lambda)` or `(mu + lambda)` survivor selection.

The starter does not choose or implement these decisions in the four student
files. Put chosen values in each file's `PARAMETERS` dictionary and explain the
exploration/exploitation reasoning in the report.

## Output

Running an implemented algorithm produces:

- one CSV containing the best candidate and final raw objective for each seed;
- one CSV containing best-so-far values every 100 objective calls;
- a median/IQR convergence plot;
- a final-objective boxplot.

Lower objective values are better. The final protocol remains 3 benchmarks x
20 final seeds = 60 runs per algorithm, all with 20,000 objective evaluations.

