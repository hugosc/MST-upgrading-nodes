__all__ = []

import numpy as np
import solution as slt

def select_candidate(sol, alpha):
	candidates = sol.vertex_impact_ratio_on_tree()
	ratio = candidates[:,0]
	d = alpha * np.ptp(ratio)
	c_min = np.min(ratio)
	return np.random.permutation(candidates[ratio <= c_min + d])[0]


def build(sol, alpha):
	s = sol.copy()
	while not s.is_saturated():
		u = select_candidate(s, alpha)
		s.upgrade_vertex(int(u[1]), update_mst=True)
	return s



def fimprov_local_search(sol, neigh, k):
	slt.first_improvement(neigh(sol), k)
	return sol




def grasp(solution, params, neigh, alpha=0.2, max_it=5, k=2):
	n_it = 0
	sol = solution(*params)
	opt = sol
	while n_it < max_it:
		# print("Grasp iteration {}".format(n_it))
		sol1 = build(sol, alpha)
		print(sol1, "rcl")
		slt.first_improvement(neigh(sol), k)
		if opt._obj_value > sol1._obj_value:
			opt = sol1
		n_it += 1
	print(opt, "opt_point")
	return opt