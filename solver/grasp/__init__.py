__all__ = []

import numpy as np
from solution import first_improvement

class Grasp:


	def __init__(self, solution, params, neigh, alpha = 0.4, max_it = 10):
		self.solution = solution
		self.params = params
		self.neigh = neigh
		self.alpha = alpha
		self.max_it = max_it


	# Select candidates in the RCS style. In this case, we want to find vertices
	#  which maximizes the ratio "sol_improv / cost".
	#
	def select_candidate(self, sol):
		candidates = sol.vertex_impact_ratio_on_tree()
		ratio = candidates[:,0]
		d = self.alpha * np.ptp(ratio)
		c_max = np.max(ratio)

		# random uniform sample of rcl
		return np.random.permutation(candidates[ratio >= c_max - d])[0]


	def build(self, sol):
		s = sol.copy()
		s.cleanse_to_state()
		while not s.is_saturated():
			u = self.select_candidate(s)
			s.upgrade_vertex(int(u[1]), update_mst=True)
		return s


	def fimprov_local_search(self, sol):
		"""BROKEN"""
		print('initial obj_value: %f' % (sol.obj_value(),))
		N = self.neigh(sol)
		while first_improvement(N, 2):
			print('improvement: %f' % (sol.obj_value(),))
			
		return sol


	# def iterated_local_search(sol, p_func, p_params, f_func, f_params, max_it=10):
	# 	n_it = 0
	# 	while n_it < max_it:
	# 		sol = 


	def run(self):
		n_it = 0
		self.solution = self.solution(*self.params)
		opt =  self.solution
		while n_it < self.max_it:
			# print("Grasp iteration {}".format(n_it))
			sol1 = self.build(self.solution)

			self.fimprov_local_search(sol1)

			if opt._obj_value > sol1.obj_value():
				opt = sol1
			n_it += 1
		print(opt)
		return opt
