__all__ = []

import numpy as np
from solution import first_improvement


# Select candidates in the RCS style. In this case, we want to find vertices
#  which maximizes the ratio "sol_improv / cost".
#
def select_candidate(sol, alpha):
	candidates = sol.vertex_impact_ratio_on_tree()
	ratio = candidates[:,0]
	d = alpha * np.ptp(ratio)
	c_max = np.max(ratio)

	# random uniform sample of rcl
	return np.random.permutation(candidates[ratio >= c_max - d])[0]


def build(sol, alpha):
	s = sol.copy()
	s.cleanse()
	while not s.is_saturated():
		u = select_candidate(s, alpha)
		s.upgrade_vertex(int(u[1]), update_mst=True)
	return s



def fimprov_local_search(sol, neigh):
	"""BROKEN"""
	N = neigh(sol)
	while first_improvement(N, 2):
		pass
		
	return sol




def grasp(solution, params, neigh, alpha=0.4, max_it=1):
	n_it = 0
	sol = solution(*params)
	opt = sol
	while n_it < max_it:
		# print("Grasp iteration {}".format(n_it))
		sol1 = build(sol, alpha)
		fimprov_local_search(sol1, neigh)

		if opt._obj_value > sol1._obj_value:
			opt = sol1
		n_it += 1
	print(opt)
	return opt