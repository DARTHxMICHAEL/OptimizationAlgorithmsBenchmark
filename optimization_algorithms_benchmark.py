import time
import numpy as np
import matplotlib.pyplot as plt


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

# OPTIMIZATION ALGORITHMS SECTION

def hill_climb_max_search(f, domain, iterations=2000, step_size=0.1, seed=0, debug=False):
	"""
	Simple stochastic hill-climbing search for a local maximum.
	"""
	rng = np.random.default_rng(seed)
	x = rng.uniform(-(domain/2), domain/2)
	y = rng.uniform(-(domain/2), domain/2)
	best_val = f(x, y)

	for _ in range(iterations):
		dx = rng.normal(scale=step_size)
		dy = rng.normal(scale=step_size)
		nx = np.clip(x + dx, -(domain/2), domain/2)
		ny = np.clip(y + dy, -(domain/2), domain/2)

		val = f(nx, ny)
		if val > best_val:
			x, y, best_val = nx, ny, val

	if(debug):
		print("\nHill Climb MAX:")
		print(f"Local MAX at (x={x:.4f}, y={y:.4f}) = {best_val:.6f}")

	return x, y, f(x, y)


def random_restart_hill_climb_max_search(f, domain, restarts=10, iterations=2000, step_size=0.1, seed=0, debug=False):
	"""
	Random-restart hill climbing for global maximization.
	"""
	rng = np.random.default_rng(seed)

	best_global = (None, None, -float("inf"))

	for r in range(restarts):
		x = rng.uniform(-(domain/2), domain/2)
		y = rng.uniform(-(domain/2), domain/2)
		best_val = f(x, y)

		for _ in range(iterations):
			dx = rng.normal(scale=step_size)
			dy = rng.normal(scale=step_size)

			nx = np.clip(x + dx, -(domain/2), domain/2)
			ny = np.clip(y + dy, -(domain/2), domain/2)

			val = f(nx, ny)
			if val > best_val:
				x, y, best_val = nx, ny, val

		if best_val > best_global[2]:
			best_global = (x, y, best_val)

	if debug:
		print("\nRandom Restart Hill Climb MAX:")
		print(f"Global MAX at (x={best_global[0]:.4f}, y={best_global[1]:.4f}) = {best_global[2]:.6f}")

	return x, y, f(x, y)


def momentum_max_search(f, domain, lr=0.01, momentum=0.9, steps=2000, seed=0, eps=1e-4, debug=False):
	"""
	Gradient-free momentum-based local maximization.
	Uses finite-difference gradient estimates.
	"""
	rng = np.random.default_rng(seed)

	x = rng.uniform(-(domain/2), domain/2)
	y = rng.uniform(-(domain/2), domain/2)

	vx, vy = 0.0, 0.0

	def estimate_grad(x, y):
		h = eps
		dx = (f(x + h, y) - f(x - h, y)) / (2*h)
		dy = (f(x, y + h) - f(x, y - h)) / (2*h)
		return dx, dy

	for _ in range(steps):
		gx, gy = estimate_grad(x, y)

		vx = momentum * vx + lr * gx
		vy = momentum * vy + lr * gy

		x = np.clip(x + vx, -(domain/2), domain/2)
		y = np.clip(y + vy, -(domain/2), domain/2)

	if(debug):
		print("\nMomentum MAX:")
		print(f"Local MAX at (x={x:.4f}, y={y:.4f}) = {f(x,y):.6f}")

	return x, y, f(x, y)


def simulated_annealing_max_search(f, domain, start_temp=1.0, end_temp=1e-3, steps=5000, step_scale=0.5, seed=0, debug=False):
	"""
	Simulated annealing search for a global maximum.
	"""
	rng = np.random.default_rng(seed)

	x = rng.uniform(-(domain/2), domain/2)
	y = rng.uniform(-(domain/2), domain/2)
	best_x, best_y = x, y
	best_val = f(x, y)

	for i in range(steps):
		t = start_temp * (1 - i/steps) + end_temp * (i/steps)

		nx = x + rng.normal(scale=step_scale * t)
		ny = y + rng.normal(scale=step_scale * t)

		nx = np.clip(nx, -(domain/2), domain/2)
		ny = np.clip(ny, -(domain/2), domain/2)

		new_val = f(nx, ny)
		old_val = f(x, y)

		if new_val > old_val:
			x, y = nx, ny
		else:
			if rng.random() < np.exp((new_val - old_val) / max(t, 1e-12)):
				x, y = nx, ny

		if new_val > best_val:
			best_x, best_y, best_val = nx, ny, new_val

	if(debug):
		print("\nSimulated Annealing MAX:")
		print(f"Local MAX at (x={best_x:.4f}, y={best_y:.4f}) = {best_val:.6f}")

	return best_x, best_y, f(best_x, best_y)


def evolution_strategy_max_search(f, domain, mu=5, lam=20, generations=200, sigma=0.5, seed=0, debug=False):
	"""
	(μ, λ) - Evolution Strategy for maximization.
	"""
	rng = np.random.default_rng(seed)

	parents = rng.uniform(
		-(domain/2), domain/2,
		size=(mu, 2)
	)

	for _ in range(generations):
		offspring = []

		for _ in range(lam):
			p = parents[rng.integers(mu)]
			child = p + rng.normal(scale=sigma, size=2)
			child = np.clip(child, -(domain/2), domain/2)
			offspring.append(child)

		offspring = np.array(offspring)
		fitness = np.array([f(x, y) for x, y in offspring])

		best_idx = np.argsort(fitness)[-mu:]
		parents = offspring[best_idx]

		sigma *= 0.99

	best_parent = max(parents, key=lambda p: f(p[0], p[1]))

	if debug:
		print("\nEvolution Strategy MAX:")
		print(f"Global MAX at (x={best_parent[0]:.4f}, y={best_parent[1]:.4f}) = {f(best_parent[0], best_parent[1]):.6f}")

	return best_parent[0], best_parent[1], f(best_parent[0], best_parent[1])


def differential_evolution_max_search(f, domain, pop_size=20, generations=300, F=0.8, CR=0.9, seed=0, debug=False):
	rng = np.random.default_rng(seed)

	pop = rng.uniform(-(domain/2), domain/2, size=(pop_size, 2))
	fitness = np.array([f(x, y) for x, y in pop])

	for _ in range(generations):
		for i in range(pop_size):
			idxs = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)
			a, b, c = pop[idxs]

			mutant = a + F * (b - c)
			mutant = np.clip(mutant, -(domain/2), domain/2)

			cross = rng.random(2) < CR
			if not np.any(cross):
				cross[rng.integers(0, 2)] = True

			trial = np.where(cross, mutant, pop[i])
			val = f(trial[0], trial[1])

			if val > fitness[i]:
				pop[i] = trial
				fitness[i] = val

	best_idx = np.argmax(fitness)
	best = pop[best_idx]

	if debug:
		print("\nDifferential Evolution MAX:")
		print(f"Global MAX at (x={best[0]:.4f}, y={best[1]:.4f}) = {fitness[best_idx]:.6f}")

	return best[0], best[1], fitness[best_idx]


def cma_es_max_search(f, domain, pop_size=20, generations=200, sigma=0.5, seed=0, debug=False):
	rng = np.random.default_rng(seed)

	mean = rng.uniform(-(domain/2), domain/2, size=2)
	cov = np.eye(2)

	for _ in range(generations):
		samples = rng.multivariate_normal(mean, cov * sigma**2, size=pop_size)
		samples = np.clip(samples, -(domain/2), domain/2)

		fitness = np.array([f(x, y) for x, y in samples])
		idx = np.argsort(fitness)[-pop_size//2:]

		elite = samples[idx]
		mean = np.mean(elite, axis=0)
		cov = np.cov(elite.T) + 1e-6 * np.eye(2)

		sigma *= 0.99

	best = max(elite, key=lambda p: f(p[0], p[1]))
	best_val = f(best[0], best[1])

	if debug:
		print("\nCMA-ES MAX:")
		print(f"Global MAX at (x={best[0]:.4f}, y={best[1]:.4f}) = {best_val:.6f}")

	return best[0], best[1], best_val


# END OF OPTIMIZATION ALGORITHMS SECTION

def run_multiple_seeds(optimizer, f, domain, runs=10):
	"""
	Run an optimizer multiple times with different seeds and summarize statistics.
	Includes per-run timing and total timing (in seconds).
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

	# Summary stats
	total = sum(values)
	avg = total / runs
	std = np.std(values)
	vmin = min(values)
	vmax = max(values)

	print("=== Multi-Run Summary ===")
	print(f"Sum:           {total:.6f}")
	print(f"Average:       {avg:.6f}")
	print(f"Std Dev:       {std:.6f}")
	print(f"Min:           {vmin:.6f}")
	print(f"Max:           {vmax:.6f}")

	print("\n=== Timing (seconds) ===")
	#print(f"Per-run avg:   {np.mean(run_times):.6f} s")
	#print(f"Per-run std:   {np.std(run_times):.6f} s")
	print(f"Fastest run:   {np.min(run_times):.6f} s")
	print(f"Slowest run:   {np.max(run_times):.6f} s")
	print(f"Total time:    {total_end - total_start:.6f} s")


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

	# hill climb optimization algorithm
	print("\n=== Hill Climb MAX ===")
	run_multiple_seeds(hill_climb_max_search, f, domain, runs)

	# hill climb with random restart optimization algorithm
	print("\n=== Hill Climb Random Restart MAX ===")
	run_multiple_seeds(random_restart_hill_climb_max_search, f, domain, runs)

	# momentum optimization algorithm
	print("\n=== Momentum MAX ===")
	run_multiple_seeds(momentum_max_search, f, domain, runs)

	# simulated annealing optimization algorithm
	print("\n=== Simulated Annealing MAX ===")
	run_multiple_seeds(simulated_annealing_max_search, f, domain, runs)

	# evolution strategy optimization algorithm
	print("\n=== Evolution Strategy MAX ===")
	run_multiple_seeds(evolution_strategy_max_search, f, domain, runs)

	# differential evolution optimization algorithm
	print("\n=== Differential Evolution MAX ===")
	run_multiple_seeds(differential_evolution_max_search, f, domain, runs)

	# cma-es optimization algorithm
	print("\n=== CMA-ES MAX ===")
	run_multiple_seeds(cma_es_max_search, f, domain, runs)


# main function execution
run_simulation(seed=123, n_terms=10, domain=50, runs=50)