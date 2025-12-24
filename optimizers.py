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