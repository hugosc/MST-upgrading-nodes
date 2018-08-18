import numpy as np
from scipy.stats import binned_statistic

from graph_tool.all import *
from graph_tool.topology import min_spanning_tree
from utils import load_instance

from itertools import product, combinations

class Solution:


	def __init__(self, budget, filepath=None, g=None):

		self.budget = budget
		self.running_cost = 0

		self._DIRTY = False
		
		assert filepath is not None or g is not None

		if filepath is not None:

			self.g = load_instance(filepath)
		else:

			self.g = g
			#self.running_cost = np.inner(g.vp.cost.get_array(), 
			#	g.vp.is_upgraded.get_array())

		assert hasattr(self.g.vp, 'is_upgraded')
		assert hasattr(self.g.vp, 'cost')
		assert hasattr(self.g.ep, 'weight')
		assert hasattr(self.g.ep, 'all_weights')

		self.N = self.g.num_vertices()

		self.ewa = self.edge_weight_array()

		self.v_cost = self.g.vp.cost.a

		self._update_mst()

		self.atualize_allowed_upgrades()


	def __str__(self):
		arr = self.g.vp.is_upgraded.get_array()
		return '{}, with obj_value {} and cost {}'.format(arr, 
			self.obj_value(), self.running_cost)


	def copy(self):
		s = Solution(self.budget, g=self.g.copy())
		s.running_cost = self.running_cost
		return s


	"""objective function value: weight of MST"""
	def obj_value(self):
		if self._DIRTY:
			self._update_mst()
			self._DIRTY = False

		return self._obj_value


	def upgrade_vertex(self, v, update_mst=False):

		g = self.g

		v = g.vertex(v)

		v_cost = g.vp.cost[v]

		if (not g.vp.is_upgraded[v]) and \
		self.running_cost + v_cost <= self.budget:

			g.vp.is_upgraded[v] = True

			for e in v.all_edges():
				
				all_weights = g.ep.all_weights[e]

				w = e.target()

				if g.vp.is_upgraded[w]:
					g.ep.weight[e] = all_weights[2]
				else:
					g.ep.weight[e] = all_weights[1] 

			self.running_cost += v_cost

			if update_mst:
				self._update_mst()
			else:
				self._DIRTY = True

			self.atualize_allowed_upgrades()

			return True
		
		return False


	def downgrade_vertex(self, v, update_mst=False):

		g = self.g

		v = g.vertex(v)

		v_cost = g.vp.cost[v]

		if g.vp.is_upgraded[v]:

			g.vp.is_upgraded[v] = False

			for e in v.all_edges():
				
				all_weights = g.ep.all_weights[e]

				w = e.target()

				if g.vp.is_upgraded[w]:
					g.ep.weight[e] = all_weights[1]
				else:
					g.ep.weight[e] = all_weights[0] 

			self.running_cost -= v_cost

			if update_mst:
				self._update_mst()
			else:
				self._DIRTY = True


			self.atualize_allowed_upgrades()

			return True	

		return False

	def upgrade_vertex_unsafe(self, v, update_mst=False):

		g = self.g

		v = g.vertex(v)

		v_cost = g.vp.cost[v]

		if not g.vp.is_upgraded[v]:

			g.vp.is_upgraded[v] = True

			for e in v.all_edges():
				
				all_weights = g.ep.all_weights[e]

				w = e.target()

				if g.vp.is_upgraded[w]:
					g.ep.weight[e] = all_weights[2]
				else:
					g.ep.weight[e] = all_weights[1] 

			self.running_cost += v_cost

			if update_mst:
				self._update_mst()
			else:
				self._DIRTY = True


			self.atualize_allowed_upgrades()

			return True
		
		return False


	def _update_mst(self):

		g = self.g

		self.mst = min_spanning_tree(g, g.ep.weight)

		self._obj_value = np.sum(self.g.ep.weight.a[self.mst.a])


	# Compute which vertices are still able to be updated
	def atualize_allowed_upgrades(self):
		self.to_upg = np.logical_and(
			np.logical_not(self.g.vp.is_upgraded.a.astype(bool)),
			self.v_cost <= (self.budget - self.running_cost))


	def available_vertices(self):
		return np.sum(self.to_upg)


	def is_saturated(self):
		return self.available_vertices() == 0


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


	# Compute current min_spam_tree upgrade cost per weight varition per vertex
	#  for all vertices simultaneously. Uses arrays as a promise of better
	#  performance from graph_tool documentation.
	#
	#  Note: iterating with iterators are slow compared to array operations.
	def vertex_impact_ratio_on_tree(self):
		upgraded = self.g.vp.is_upgraded.a.astype(bool)

		on_mst = self.mst.a.astype(bool)
		edges = self.g.get_edges()[on_mst]
		weights = self.g.ep.weight.a[on_mst]

		e_to_upg_1 = self.to_upg[edges[:,0]]
		e_to_upg_2 = self.to_upg[edges[:,1]]

		e_indexes = np.concatenate((edges[e_to_upg_1, 2], edges[e_to_upg_2, 2]))

		# uses undirectedness of edges to represent different upgraded vertices
		edges = np.concatenate((edges[:,:2][e_to_upg_1], 
			edges[:,-2::-1][e_to_upg_2]))
		weights = np.concatenate((weights[e_to_upg_1], weights[e_to_upg_2]))
		delta = weights - self.ewa[e_indexes,
		 self.to_upg[edges[:, 0]].astype(int) + upgraded[edges[:, 1]]]

		# graph undirected -> edges needs to be accounted for both vertices
		delta = binned_statistic(
			edges[:,0],
			delta,
			statistic=np.sum,
			bins=np.append(np.arange(self.N)[self.to_upg], [self.N + 1]))

		# maybe pure python will prove not effective
		# how much you spent per upgrade unit
		# print(np.column_stack((self.v_cost[self.to_upg] / delta[0], 
		# 		delta[1][:-1])))
		return np.column_stack((self.v_cost[self.to_upg] / delta[0], 
				delta[1][:-1]))


class Neighbourhood:

	def __init__(self, s):
		self.s = s

	def non_sub_bits(bits):
		"""NOT USED ANYMORE, possibly will be deleted"""

		for s in product([False, True], repeat=len(bits)):

			if any(map(lambda b: b[0] and (not b[1]), zip(s, bits))):
				yield s


	def neighbour_codes(self, k):

		s = self.s

		for comb in combinations(range(s.N), k):

			g = s.g

			delta = 0
			exists_false = False
			for v in comb:

				if g.vp.is_upgraded[v]:
					delta -= g.vp.cost[v]

				else:
					exists_false = True
					delta += g.vp.cost[v]

			if s.running_cost + delta <= s.budget:

				yield comb

	def try_update(self, update_code):

		s = self.s

		for v in update_code:

			if not s.g.vp.is_upgraded[v]:
				s.upgrade_vertex_unsafe(v)
			else:
				s.downgrade_vertex(v)

		obj_value = s.obj_value()

		for v in update_code:

			if not s.g.vp.is_upgraded[v]:
				s.upgrade_vertex_unsafe(v)
			else:
				s.downgrade_vertex(v)

		return obj_value

	def update(self, update_code):

		s = self.s
 
		for v in update_code:

			if not s.g.vp.is_upgraded[v]:
				s.upgrade_vertex_unsafe(v)
			else:
				s.downgrade_vertex(v)

		return s.obj_value()

def first_improvement(N, k):

	current_v = N.s.obj_value()
	print("testing first_improvement")

	for c in N.neighbour_codes(k):

		v = N.try_update(c)

		if v < current_v:

			N.update(c)
			print(N.s, "soll")
			return True

	return False