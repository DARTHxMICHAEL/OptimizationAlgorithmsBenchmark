import time
import numpy as np
import matplotlib.pyplot as plt

from optimizers import (
    hill_climb_max_search,
    random_restart_hill_climb_max_search,
    momentum_max_search,
    simulated_annealing_max_search,
    evolution_strategy_max_search,
    differential_evolution_max_search,
    cma_es_max_search,
)


def generate_random_surface(seed, n_terms, domain):
	"""
	Generate a 3D random Fourier-based surface function f(x, y).

	Parameters
	----------
	seed : int
		Deterministic seed controlling amplitudes, frequencies, and phases.
	n_terms : int
		Number of Fourier terms (controls complexity of the surface).
	domain : float
		Width/height of the square domain centered at 0.
		Domain ranges from [-domain/2, +domain/2].

	Returns
	-------
	f : function
		Callable function f(x, y) -> float that computes the surface value.
	"""
	rng = np.random.default_rng(seed)

	# random coefficients
	amps = rng.normal(scale=1.0, size=(n_terms,))
	freq_x = rng.uniform(0.5, 3.0, size=(n_terms,))
	freq_y = rng.uniform(0.5, 3.0, size=(n_terms,))
	phases = rng.uniform(0, 2*np.pi, size=(n_terms,))

	def f(x, y):
		"""
		Evaluate the generated surface at coordinates (x, y).
		"""
		base = domain/(domain*150) * (x**2 + y**2)

		noise = sum(
			amps[i] * np.sin(freq_x[i]*x + freq_y[i]*y + phases[i])
			for i in range(n_terms)
		)
		return base + noise

	return f


def brute_force_min_max_search(f, domain):
	"""
	Simple brute force search for min and max value using dynamic value step.
	"""
	step = domain/100
	brute_xs = np.arange(-(domain/2), domain/2, step)
	brute_ys = np.arange(-(domain/2), domain/2, step)

	global_min = (None, None, float("inf"))   # (x, y, value)
	global_max = (None, None, -float("inf"))   # (x, y, value)

	for x in brute_xs:
		for y in brute_ys:
			val = f(x, y)

			if val < global_min[2]:
				global_min = (x, y, val)

			if val > global_max[2]:
				global_max = (x, y, val)

	print("\n==== Brute Force Global Search Results ====")
	print(f"Global MIN at (x={global_min[0]:.4f}, y={global_min[1]:.4f}) = {global_min[2]:.6f}")
	print(f"Global MAX at (x={global_max[0]:.4f}, y={global_max[1]:.4f}) = {global_max[2]:.6f}")


def run_multiple_seeds(optimizer, f, domain, runs=10):
	"""
	Run an optimizer multiple times with different seeds and summarize statistics.
	Returns a dictionary with aggregated performance and timing metrics.
	"""
	values = []
	run_times = []

	total_start = time.perf_counter()

	for seed in range(runs):
		t0 = time.perf_counter()
		_, _, best_val = optimizer(f, domain, seed=seed)
		t1 = time.perf_counter()

		values.append(best_val)
		run_times.append(t1 - t0)

	total_end = time.perf_counter()

	total = float(np.sum(values))
	avg = float(np.mean(values))
	std = float(np.std(values))
	vmin = float(np.min(values))
	vmax = float(np.max(values))

	fastest = float(np.min(run_times))
	slowest = float(np.max(run_times))
	total_time = float(total_end - total_start)

	print("=== Multi-Run Summary ===")
	print(f"Sum:           {total:.6f}")
	print(f"Average:       {avg:.6f}")
	print(f"Std Dev:       {std:.6f}")
	print(f"Min:           {vmin:.6f}")
	print(f"Max:           {vmax:.6f}")

	print("\n=== Timing (seconds) ===")
	print(f"Fastest run:   {fastest:.6f} s")
	print(f"Slowest run:   {slowest:.6f} s")
	print(f"Total time:    {total_time:.6f} s")

	return {
		"sum": total,
		"avg": avg,
		"std": std,
		"min": vmin,
		"max": vmax,
		"fastest": fastest,
		"slowest": slowest,
		"total_time": total_time
	}


def run_optimizer_benchmark(name, optimizer, f, domain, runs, results):
	"""
	Run a single optimization algorithm benchmark and store results.
	"""
	print(f"\n=== {name} MAX ===")
	results[name] = run_multiple_seeds(optimizer, f, domain, runs)


def print_final_comparison(results):
	"""
	Print a consolidated comparison table and a ranked composite score
	across all optimization algorithms.
	"""
	print("\n================ FINAL ALGORITHM COMPARISON ================\n")

	header = f"{'Algorithm':<28} {'Avg':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Total Time (s)':>16}"
	print(header)
	print("-" * len(header))

	for name, r in results.items():
		print(
			f"{name:<28} "
			f"{r['avg']:>10.4f} "
			f"{r['std']:>10.4f} "
			f"{r['min']:>10.4f} "
			f"{r['max']:>10.4f} "
			f"{r['total_time']:>16.4f}"
		)

	# ---------------- composite score computation ----------------

	# extract ranges for normalization
	avgs = [r["avg"] for r in results.values()]
	maxs = [r["max"] for r in results.values()]
	mins = [r["min"] for r in results.values()]
	stds = [r["std"] for r in results.values()]
	times = [r["total_time"] for r in results.values()]

	max_avg = max(avgs)
	max_max = max(maxs)
	max_min = max(mins)
	max_std = max(stds)
	max_time = max(times)

	scored = []

	for name, r in results.items():
		# quality metrics (higher is better)
		avg_score = r["avg"] / max_avg
		max_score = r["max"] / max_max
		min_score = r["min"] / max_min

		# stability metric (lower is better)
		cv = r["std"] / max(r["avg"], 1e-12)
		stability_score = 1.0 - min(cv, 1.0)

		# time metric (lower is better)
		time_score = 1.0 - (r["total_time"] / max_time)

		composite = (
			0.45 * avg_score +
			0.15 * max_score +
			0.10 * min_score +
			0.15 * stability_score +
			0.15 * time_score
		)

		scored.append((name, composite, avg_score, stability_score, time_score))

	scored.sort(key=lambda x: x[1], reverse=True)

	print("\n================ RANKED COMPOSITE SCORE ====================")
	for rank, (name, score, q, s, t) in enumerate(scored, start=1):
		print(
			f"{rank:>2}. {name:<25} "
			f"Score={score:.4f} "
			f"(Quality={q:.3f}, Stability={s:.3f}, Time={t:.3f})"
		)


def run_simulation(seed=0, n_terms=5, domain=50, step=0.01, runs=10):
	"""
	Creates a random surface, visualizes it, and performs a brute-force
	global min/max search over the designated domain.

	Parameters
	----------
	seed : int
		Seed used to generate deterministic random surface.
	n_terms : int
		Number of Fourier terms creating fluctuations.
	domain : float
		Width/height of the square domain centered at 0.
		Domain ranges from [-domain/2, +domain/2].
	step : float, optional
		Step size used in brute-force global searching across the domain.
		Smaller step = more accurate but slower scan.

	Returns
	-------
	result : dict
		Contains global min/max values and their coordinates.
	"""

	f = generate_random_surface(seed, n_terms, domain)

	xs = np.linspace(-(domain/2), domain/2, 150)
	ys = np.linspace(-(domain/2), domain/2, 150)
	X, Y = np.meshgrid(xs, ys)
	Z = f(X, Y)

	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X, Y, Z)
	ax.set_title("Random 3D Function Surface")
	plt.show()

	brute_force_min_max_search(f, domain)

	results = {}

	run_optimizer_benchmark(
		"Hill Climb",
		hill_climb_max_search,
		f, domain, runs, results
	)

	run_optimizer_benchmark(
		"Hill Climb Random Restart",
		random_restart_hill_climb_max_search,
		f, domain, runs, results
	)

	run_optimizer_benchmark(
		"Momentum",
		momentum_max_search,
		f, domain, runs, results
	)

	run_optimizer_benchmark(
		"Simulated Annealing",
		simulated_annealing_max_search,
		f, domain, runs, results
	)

	run_optimizer_benchmark(
		"Evolution Strategy",
		evolution_strategy_max_search,
		f, domain, runs, results
	)

	run_optimizer_benchmark(
		"Differential Evolution",
		differential_evolution_max_search,
		f, domain, runs, results
	)

	run_optimizer_benchmark(
		"CMA-ES",
		cma_es_max_search,
		f, domain, runs, results
	)

	print_final_comparison(results)


# main function execution
run_simulation(seed=123, n_terms=10, domain=50, runs=50)