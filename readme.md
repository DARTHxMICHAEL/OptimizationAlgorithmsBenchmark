# Optimization Algorithms Benchmark (Python)

The main aim of this program is to create fully deterministic envirement and testing procedure for many
different optimization algoritms.

Source: [DARTHxMICHAEL/TabuSearchVRP](https://github.com/DARTHxMICHAEL/TabuSearchVRP)

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

 
