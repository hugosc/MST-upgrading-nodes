from solution import *
import numpy as np

def gen_viables(budget, costs):

	n = len(costs)

	solutions = set([tuple([False]* n)])
	maximals = set()

	while len(solutions):

		v = solutions.pop()

		is_maximal = True

		for i in range(n):

			if not v[i]:

				u = list(v)
				u[i] = True

				if np.inner(u, costs) <= budget:

					is_maximal = False

					solutions.add(tuple(u))

		if is_maximal:

			if v not in maximals:
				yield np.array(v)
				maximals.add(v)

def perturbate_and_local_search(sol, n_actives, n_inactives, n_total):
	
	pert = sol.random_perturbation_candidates(
		n_actives, n_inactives, n_total)

	for v in pert:
		sol.downgrade_vertex(v)

	sol._update_mst()
	base_sol = sol.upgraded.copy()
	base_mst = sol.mst.copy()

	costs = sol.globals.v_cost[pert]
	budget = sol.globals.budget - sol.running_cost

	initial_rc = sol.running_cost

	best_v = 1e10
	best_upgrade = None

	print('budget and costs', budget, costs)

	for viable in gen_viables(budget, costs):

		upgrade = pert[viable].astype(int)

		print('vertices to upgrade: ', upgrade)

		sol.batch_vertex_upgrade(upgrade, update_mst=True)

		if sol.obj_value() < best_v:

			best_v = sol.obj_value()
			best_upgrade = upgrade

		sol.cleanse_to_state(base_sol.copy(), mst=base_mst)

		assert initial_rc == sol.running_cost

	if best_upgrade is not None:
		sol.batch_vertex_upgrade(best_upgrade, update_mst=True)

	return sol.obj_value()





