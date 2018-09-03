__all__ = []

import numpy as np
from solution import first_improvement, Neighbourhood
import time



class Grasp:


	def __init__(self, solution, params, neigh, alpha = 0.12, max_it = 10,
		max_elite_size = 3):
		self.solution = solution
		self.params = params
		self.neigh = neigh
		self.alpha = alpha
		self.max_it = max_it
		self.max_elite_size = max_elite_size
		self.elite = []

	# Select candidates in the RCS style. In this case, we want to find vertices
	#  which maximizes the ratio "sol_improv / cost".
	#
	def select_candidate(self, sol):
		candidates = sol.vertex_impact_ratio_on_tree()
		ratio = candidates[:,0]
		alpha = np.min(
			[1, np.max([0, np.random.normal(loc=self.alpha, scale = 0.1)])])
		d = alpha * np.ptp(ratio)
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

	def solution_distance(self, sol1, sol2):
		return np.count_nonzero(sol1.upgraded!=sol2.upgraded)

	def path_rel(self, sol):
		lim = np.log2(sol.globals.N)
		if not self.elite: #is empty
			self.elite.append(sol)
			return
		for s in self.elite:
			s_aux = sol.copy()
			s_best = sol.copy()
			to_downgrade = np.arange(s.globals.N)[np.logical_and(
								sol.upgraded,np.logical_not(s.upgraded))]
			to_upgrade = np.arange(s.globals.N)[np.logical_and(
								s.upgraded,np.logical_not(sol.upgraded))]
			while to_upgrade.size > 0:
				n = Neighbourhood(s_aux)
				best_ind = np.argmin(
							[n.try_update((v,)) for v in to_upgrade])
				s_aux.upgrade_vertex_unsafe(to_upgrade[best_ind])
				to_upgrade = np.concatenate(
							(to_upgrade[:best_ind],to_upgrade[best_ind+1:])) 
				
				while s_aux.running_cost > s_aux.globals.budget:
					best_ind = np.argmax([
								n.try_update((v,)) for v in to_downgrade])
					s_aux.downgrade_vertex(to_downgrade[best_ind])
					to_downgrade = np.concatenate(
							(to_downgrade[:best_ind],to_downgrade[best_ind+1:]))
				if s_aux.obj_value() < s_best.obj_value():
					s_best = s_aux.copy()

				#print('dist',self.solution_distance(s_aux,s))
				#print('aux',s_aux )
			#print('best',s_best )
			if len(self.elite) < self.max_elite_size:
				self.elite.append(s_best)
			else:
				similar = False
				for i in range(len(self.elite)):
					if(self.solution_distance(self.elite[i],s_best) < lim):
						similar = True
				if(not similar):
					for i in range(len(self.elite)):
						if(s_best.obj_value() < self.elite[i].obj_value()):
							self.elite[i] = s_best.copy()
							break 


	def run(self, local_search, *params):
		n_it = 0
		self.solution = self.solution(*self.params)
		opt =  self.solution
		t = time.time()
		while n_it < self.max_it:
			# print("Grasp iteration {}".format(n_it))
			sol1 = self.build(self.solution)
			print("construção: ", sol1.obj_value())
			local_search(sol1, *params)
			print("melhoria ils: ", sol1.obj_value())

			if opt._obj_value > sol1.obj_value():
				opt = sol1

			print(opt)

			print('PR+++++++')
			self.path_rel(opt)
			print('PR-------')

			for s in self.elite: print('elite',s)	

			n_it += 1
			# print(time.time() - t1)	
		return opt, time.time() - t


def ils(sol, n_actives, n_inactives, n_total, n_it):
	for _ in range(n_it):
		perturbate_and_local_search(sol, n_actives, n_inactives, n_total)
	return sol.obj_value()


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

	# print('budget and costs', budget, costs)

	for viable in gen_viables(budget, costs):

		upgrade = pert[viable].astype(int)

		# print('vertices to upgrade: ', upgrade)

		sol.batch_vertex_upgrade(upgrade, update_mst=True)

		if sol.obj_value() < best_v:

			best_v = sol.obj_value()
			best_upgrade = upgrade

		sol.cleanse_to_state(base_sol.copy(), mst=base_mst)

		assert initial_rc == sol.running_cost

	if best_upgrade is not None:
		sol.batch_vertex_upgrade(best_upgrade, update_mst=True)

	return sol.obj_value()
