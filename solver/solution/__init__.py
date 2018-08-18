import numpy as np
from scipy.stats import binned_statistic

from graph_tool.all import *
from graph_tool.topology import min_spanning_tree
from utils import load_instance

class SolutionGlobals:

	# everything that must be done ONLY once
	def __init__(self, budget, filepath=None, g=None):

		self.budget = budget
		self.running_cost = 0
		
		assert filepath is not None or g is not None

		if filepath is not None:

			self.g = load_instance(filepath)
		else:
			self.g = g

		assert hasattr(self.g.vp, 'is_upgraded')
		assert hasattr(self.g.vp, 'cost')
		assert hasattr(self.g.ep, 'weight')
		assert hasattr(self.g.ep, 'all_weights')

		self.N = self.g.num_vertices()
		self.E = self.g.num_edges()
		self.ewa = self.edge_weight_array()
		self.v_cost = self.g.vp.cost.a
		self.edges = self.g.get_edges()


	# Gen array of edge weights with dimension (e,3).
	#  Note: duplicate information from edges to make it faster
	def edge_weight_array(self):
		ewa = np.zeros((2 * self.g.num_edges(), 3))
		edges = self.g.get_edges()
		e_len = self.g.num_edges()
		i = 0
		for e in edges:
			edge = self.g.edge(e[0], e[1])
			ewa[i] = np.array(self.g.ep.all_weights[edge])
			ewa[i + e_len] = np.array(self.g.ep.all_weights[edge])
			i += 1
		return ewa


class Solution:


	def __init__(self, budget=None, filepath=None, g=None, sol_globals=None, 
				 sol_state=None):

		# everything that must be done ONLY once goes to globals
		if sol_globals is not None:
			self.globals = sol_globals
		else:
			self.globals = SolutionGlobals(budget, filepath, g)

		if sol_state is not None:
			self.upgraded = sol_state
		else:
			self.upgraded = self.globals.g.vp.is_upgraded.a

		self.running_cost = np.inner(self.globals.v_cost,self.upgraded)
		self.cur_edge_weight = self.get_edge_weights()
		self.edge_upgrade_level = np.zeros(len(self.cur_edge_weight), dtype=int)

		self._update_mst()
		self.atualize_allowed_upgrades()


	# Mix local information of upgraded vertices with global weight weights to
	#  find current edge weight values.
	def get_edge_weights(self):
		edges = self.globals.edges
		return self.globals.ewa[np.arange(self.globals.E, dtype=int),
		self.upgraded[edges[:, 0]].astype(int) + self.upgraded[edges[:, 1]]]


	def cleanse(self):
		self.upgraded = np.zeros(self.globals.N, dtype=bool)
		self.running_cost = 0
		self.cur_edge_weight = self.get_edge_weights()
		self.edge_upgrade_level = np.zeros(len(self.cur_edge_weight), dtype=int)
		self._update_mst()
		self.atualize_allowed_upgrades()

	def __str__(self):
		arr = self.upgraded
		return '{}, with obj_value {}'.format(arr, self.obj_value())


	def copy(self):
		return Solution(sol_globals=self.globals, sol_state=self.upgraded)


	"""objective function value: weight of MST"""
	def obj_value(self):
		return self._obj_value


	# Assumes you know what you're doing. Performs both upgrade and downgrade
	#  using mode selection.
	def fast_v_upgrade(self, v, mode=True):
		inc_mult = (mode * 1) + (not mode * -1)
		v_cost = self.globals.v_cost[v]

		self.fast_weight_update(v, inc_mult)

		self.upgraded[v] = mode
		self.running_cost += inc_mult * v_cost

		self._update_mst()
		self.atualize_allowed_upgrades()


	def fast_weight_update(self, v, inc=1):
		e_indexes = self.globals.edges[:, 2][np.logical_or(
			self.globals.edges[:,0] == v,
			self.globals.edges[:, 1] == v)]
		self.edge_upgrade_level[e_indexes] += inc
		self.cur_edge_weight[e_indexes] = self.globals.ewa[e_indexes, 
			self.edge_upgrade_level[e_indexes]]



	def upgrade_vertex(self, v):

		if not self.upgraded[v] and \
		self.running_cost + self.globals.v_cost[v] <= self.globals.budget:
			self.fast_v_upgrade(v)
			return True

		return False

	def downgrade_vertex(self, v):

		if self.upgraded[v]:
			self.fast_v_upgrade[v, False]
			return True

		return False


	def _update_mst(self):

		g = self.globals.g
		g.ep.weight.a = self.cur_edge_weight
		self.mst = min_spanning_tree(g, g.ep.weight)

		self._obj_value = sum(
			map(lambda e: g.ep.weight[e] * self.mst[e], g.edges()))


	# Compute which vertices are still able to be updated
	def atualize_allowed_upgrades(self):
		self.to_upg = np.logical_and(
			np.logical_not(self.upgraded),
			self.globals.v_cost <= (self.globals.budget - self.running_cost))


	def available_vertices(self):
		return np.sum(self.to_upg)


	def is_saturated(self):
		return self.available_vertices() == 0



	# Compute current min_spam_tree upgrade cost per weight varition per vertex
	#  for all vertices simultaneously. Uses arrays as a promise of better
	#  performance from graph_tool documentation.
	#
	#  Note: iterating with iterators are slow compared to array operations.
	def vertex_impact_ratio_on_tree(self):

		on_mst = self.mst.a.astype(bool)
		edges = self.globals.g.get_edges()[on_mst]
		weights = self.cur_edge_weight[on_mst]

		e_to_upg_1 = self.to_upg[edges[:,0]]
		e_to_upg_2 = self.to_upg[edges[:,1]]

		e_indexes = np.concatenate((edges[e_to_upg_1, 2], edges[e_to_upg_2, 2]))

		# uses undirectedness of edges to represent different upgraded vertices
		edges = np.concatenate((edges[:,:2][e_to_upg_1], 
			edges[:,-2::-1][e_to_upg_2]))
		weights = np.concatenate((weights[e_to_upg_1], weights[e_to_upg_2]))
		delta = weights - self.globals.ewa[e_indexes,
		 self.to_upg[edges[:, 0]].astype(int) + self.upgraded[edges[:, 1]]]

		# graph undirected -> edges needs to be accounted for both vertices
		delta = binned_statistic(
			edges[:,0],
			delta,
			statistic=np.sum,
			bins=np.append(np.arange(self.globals.N)[self.to_upg],
				[self.globals.N + 1]))

		# maybe pure python will prove not effective
		# how much you spent per upgrade unit
		# print(np.column_stack((self.v_cost[self.to_upg] / delta[0], 
		# 		delta[1][:-1])))
		return np.column_stack((self.globals.v_cost[self.to_upg] / delta[0], 
				delta[1][:-1]))



class _NeighbourhoodIterator:


	def __init__(self, s, neigh, i=-1):
		
		self.s = s
		self.neigh = neigh
		self.i = i

		self.count = 0


	def __next__(self):

		if self.i < self.s.globals.N:

			if self.i >= 0:
				self.s.downgrade_vertex(self.i)

			self.i += 1

			while self.i < self.s.globals.N:
				print("iterating 1")
				if self.s.upgrade_vertex(self.i):
					print("iterating 2")
					self.neigh._mem[self.count] = self.i
					self.count += 1
					return self.s

				self.i += 1

		raise StopIteration


class Neighbourhood:


	"""Very basic neighbourhood for validation purposes"""
	def __init__(self, s):
		self.s = s
		self._mem = [-1] * s.globals.N


	def __iter__(self):
		return _NeighbourhoodIterator(self.s.copy(), self)


	def __getitem__(self, i):
		"""returns i-th neighbour
		if not already calculated, 
		then the iterator is called until neighbour is obtained
		"""
		if i < 0 or i >= self.s.globals.N:
			raise KeyError
		
		if self._mem[i] != -1:

			s_ = self.s.copy()
			s_.upgrade_vertex(i)

			return s_
		else:
			it = iter(self)

			while it.count < i:
				next(it)

			return next(it).copy()