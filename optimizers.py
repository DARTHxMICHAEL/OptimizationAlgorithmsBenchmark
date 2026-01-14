def linear_gradient_max_search(f, domain, seed=0, debug=False, iterations=500, lr=0.05, eps=1e-4):
	"""
	Linear (first-order) maximization using finite - difference gradient ascent.
	"""

	rng = np.random.default_rng(seed)

	x = rng.uniform(-(domain / 2), domain / 2)
	y = rng.uniform(-(domain / 2), domain / 2)

	for _ in range(iterations):
		fx = f(x, y)

		# finite-difference gradients
		dfdx = (f(x + eps, y) - fx) / eps
		dfdy = (f(x, y + eps) - fx) / eps

		x += lr * dfdx
		y += lr * dfdy

		x = np.clip(x, -(domain / 2), domain / 2)
		y = np.clip(y, -(domain / 2), domain / 2)

	best_val = f(x, y)

	if debug:
		print("\nLinear Gradient MAX:")
		print(f"Local MAX at (x={x:.4f}, y={y:.4f}) = {best_val:.6f}")

	return x, y, best_val


def ahp_max_search(f, domain, seed=0, debug=False, samples=25, preference_eps=1e-6):
	"""
	Analytical Hierarchy Process - inspired maximization using discrete sampling and pairwise dominance.
	"""

	rng = np.random.default_rng(seed)

	# sample candidate points
	points = rng.uniform(
		-(domain / 2), domain / 2,
		size=(samples, 2)
	)

	values = np.array([f(x, y) for x, y in points])

	# pairwise comparison matrix
	A = np.ones((samples, samples))

	for i in range(samples):
		for j in range(samples):
			if i != j:
				A[i, j] = (values[i] + preference_eps) / (values[j] + preference_eps)

	# principal eigenvector (priority vector)
	eigvals, eigvecs = np.linalg.eig(A)
	max_idx = np.argmax(np.real(eigvals))
	priorities = np.real(eigvecs[:, max_idx])
	priorities /= priorities.sum()

	best_idx = np.argmax(priorities)
	x_best, y_best = points[best_idx]
	best_val = values[best_idx]

	if debug:
		print("\nAHP-based MAX:")
		print(f"Best candidate (x={x_best:.4f}, y={y_best:.4f}) = {best_val:.6f}")

	return x_best, y_best, best_val


def graph_greedy_max_search(f, domain, seed=0, debug=False, grid_size=50, max_steps=500):
	"""
	Graph - based maximization using greedy ascent on a discretized grid graph.
	"""

	rng = np.random.default_rng(seed)

	# create grid
	xs = np.linspace(-(domain / 2), domain / 2, grid_size)
	ys = np.linspace(-(domain / 2), domain / 2, grid_size)

	# random start node
	i = rng.integers(0, grid_size)
	j = rng.integers(0, grid_size)

	def neighbors(i, j):
		for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
					   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
			ni, nj = i + di, j + dj
			if 0 <= ni < grid_size and 0 <= nj < grid_size:
				yield ni, nj

	current_val = f(xs[i], ys[j])

	for _ in range(max_steps):
		best_i, best_j = i, j
		best_val = current_val

		for ni, nj in neighbors(i, j):
			val = f(xs[ni], ys[nj])
			if val > best_val:
				best_i, best_j = ni, nj
				best_val = val

		if best_val <= current_val:
			break

		i, j = best_i, best_j
		current_val = best_val

	x_best = xs[i]
	y_best = ys[j]

	if debug:
		print("\nGraph Greedy MAX:")
		print(f"Graph MAX at (x={x_best:.4f}, y={y_best:.4f}) = {current_val:.6f}")

	return x_best, y_best, current_val
	

def hill_climb_max_search(f, domain, iterations=4000, step_size=0.1, seed=0, debug=False):
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


def random_restart_hill_climb_max_search(f, domain, restarts=10, iterations=400, step_size=0.1, seed=0, debug=False):
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


def momentum_max_search(f, domain, lr=0.01, momentum=0.9, steps=4000, seed=0, eps=1e-4, debug=False):
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


def simulated_annealing_max_search(f, domain, start_temp=1.0, end_temp=1e-3, steps=4000, step_scale=0.5, seed=0, debug=False):
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


def differential_evolution_max_search(f, domain, pop_size=20, generations=200, F=0.8, CR=0.9, seed=0, debug=False):
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

	if pop_size < 4:
		raise ValueError("CMA-ES requires pop_size >= 4")

	rng = np.random.default_rng(seed)

	dim = 2
	bounds = domain / 2

	# --- strategy parameters ---
	lam = pop_size
	mu = lam // 2

	weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
	weights /= np.sum(weights)
	mu_eff = 1 / np.sum(weights**2)

	# --- learning rates ---
	c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
	d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
	c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
	c1 = 2 / ((dim + 1.3)**2 + mu_eff)
	c_mu = min(
		1 - c1,
		2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)
	)

	# --- initialization ---
	mean = rng.uniform(-bounds, bounds, size=dim)
	C = np.eye(dim)
	p_sigma = np.zeros(dim)
	p_c = np.zeros(dim)

	eigvals = np.ones(dim)
	B = np.eye(dim)
	D = np.ones(dim)
	inv_sqrt_C = np.eye(dim)

	chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

	best = None
	best_val = -np.inf

	for gen in range(generations):
		# --- sample population ---
		z_samples = rng.standard_normal((lam, dim))
		samples = mean + sigma * (z_samples @ (B * D).T)

		# boundary repair (projection)
		samples = np.clip(samples, -bounds, bounds)

		samples = np.array(samples)
		z_samples = np.array(z_samples)

		# --- evaluate fitness ---
		fitness = np.array([f(x[0], x[1]) for x in samples])
		idx = np.argsort(fitness)[-mu:]

		elite = samples[idx]
		z_elite = z_samples[idx]

		# --- track currently best ---
		if fitness[idx[-1]] > best_val:
			best_val = fitness[idx[-1]]
			best = elite[-1]

		old_mean = mean.copy()

		# --- recombination ---
		mean = np.sum(elite * weights[:, None], axis=0)
		z_mean = np.sum(z_elite * weights[:, None], axis=0)

		# --- step-size adaptation (with overflow preventing clamp) ---
		p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ z_mean)

		exponent = (np.linalg.norm(p_sigma) / chi_n - 1) * c_sigma / d_sigma
		exponent = np.clip(exponent, -5.0, 5.0)
		sigma *= np.exp(exponent)

		# --- prevent sigma outside the domain
		sigma = np.clip(sigma, 1e-8, bounds)

		# --- covariance adaptation ---
		h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
		p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

		rank_one = np.outer(p_c, p_c)
		rank_mu = np.zeros((dim, dim))
		for i in range(mu):
			diff = (elite[i] - old_mean) / sigma
			rank_mu += weights[i] * np.outer(diff, diff)

		C = (
			(1 - c1 - c_mu) * C
			+ c1 * rank_one
			+ c_mu * rank_mu
		)

		# --- eigen decomposition (periodic) ---
		if gen % 10 == 0:
			eigvals, B = np.linalg.eigh(C)
			eigvals = np.maximum(eigvals, 1e-12)
			D = np.sqrt(eigvals)
			inv_sqrt_C = B @ np.diag(1 / D) @ B.T

	if debug:
		print("\nCMA-ES MAX:")
		print(f"Global MAX at (x={best[0]:.4f}, y={best[1]:.4f}) = {best_val:.6f}")

	return best[0], best[1], best_val