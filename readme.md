# Optimization Algorithms Benchmark (Python)

The main aim of this program is to create fully deterministic envirement and testing procedure for many
different optimization algoritms.

Source: [DARTHxMICHAEL/OptimizationAlgorithmsBenchmark](https://github.com/DARTHxMICHAEL/OptimizationAlgorithmsBenchmark)

## Instalation and basic use

Dependencies:
```bash
pip install numpy
pip install matplotlib
```

Example usage:

```
run_simulation(seed=0, n_terms=5, domain=50, step=0.01, runs=10)
```

## Custom optimization algorithm addition

In order to add custom optimization algorithm add the following lines to the main **run_simulation** function.
```
# your optimization algorithm
print("\n=== Your Optimization Algorithm MAX ===")
run_multiple_seeds(your_optimization_algorithm, f, domain, runs)
```

Make sure to take **function** and **domain** as optimization function params and return the triplet of **x**, **y** of the best value and the value itself **best_val**.

```
def your_optimization_algorithm(f, domain, ...):
    rng = np.random.default_rng(seed)

    x = rng.uniform(-(domain/2), domain/2)
    y = rng.uniform(-(domain/2), domain/2)

   ...

    return x, y, f(x, y)
```

## Code explanation

In order to test optimization function we create fully deterministic 3D function withing the designated <X,Y> domain. The gnerated surface is defined as **f(x, y) = base(x, y) + Σ a_i · sin(fx_i · x + fy_i · y + φ_i)**, where the domain is the base function we fluctuate here. In our case the base function is representedas as **base(x, y) = (1 / 150) · (x² + y²)**, where the function is defined over a square domain **x, y ∈ [−domain / 2, +domain / 2]**.

Later we use **run_multiple_seeds** function that allow us to run multiple iterations of tests with different seeds each time, after which we summerize basic params such as sum, average, standard deviation or run times. This allows us to summerize and compare different optimization algoritmhs within the repeatable (deterministic) envirement.


## List of optimization algorithms

The list of implemented optimization algorithms include:

- **Hill Climbing (hill_climb_max_search)** is a simple local optimization method that iteratively moves toward better solutions by applying small random perturbations and accepting only improving steps. It is computationally inexpensive and easy to implement but is highly sensitive to the initial starting point. Because it never accepts worse solutions, it often converges quickly to a local maximum. As a result, it performs poorly on multimodal or noisy landscapes. It is primarily useful as a baseline method.

- **Random Restart Hill Climbing (random_restart_hill_climb_max_search)** extends basic hill climbing by running multiple independent searches from different random starting points. This increases the probability of discovering better local optima across the search space. The method retains the simplicity and speed of hill climbing while significantly improving global exploration. It is still probabilistic and does not guarantee a global optimum, but it provides a strong performance-to-cost ratio. This approach is often effective for moderately complex landscapes.

- **Momentum-based (momentum_max_search)** optimization augments gradient ascent by accumulating a velocity vector that smooths updates across iterations. This helps accelerate convergence in consistent ascent directions and reduces oscillations. In this implementation, gradients are estimated numerically using finite differences, which increases computational cost. While momentum improves stability on smooth surfaces, it remains a local method and is still vulnerable to local optima. Its effectiveness depends strongly on learning rate and momentum parameters.

- **Simulated Annealing (simulated_annealing_max_search)** is a probabilistic global optimization algorithm inspired by the physical annealing process. It allows occasional acceptance of worse solutions according to a temperature-controlled probability, enabling escape from local maxima. Early in the search, exploration is emphasized, while later iterations focus on exploitation as the temperature decreases. This makes the algorithm well suited for rugged, multimodal landscapes. Performance depends on the cooling schedule and step-size parameters.

- **Evolution Strategies (evolution_strategy_max_search)** are population-based stochastic optimization methods that iteratively improve candidate solutions through mutation and selection. A set of parent solutions generates offspring via random perturbations, and the best-performing individuals are retained. This approach is robust to noise, nonlinearity, and multimodality, and does not require gradient information. Evolution Strategies tend to be more computationally expensive but offer strong global search capabilities. They are widely used in continuous optimization problems where gradients are unreliable or unavailable.
 
