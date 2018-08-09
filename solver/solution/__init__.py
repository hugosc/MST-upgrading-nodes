import numpy as np
from scipy.stats import binned_statistic

from graph_tool.all import *
from graph_tool.topology import min_spanning_tree
from utils import load_instance

class Solution:


	def __init__(self, budget, filepath=None, g=None):

		self.budget = budget
		self.running_cost = 0
		
		assert filepath is not None or g is not None

		if filepath is not None:

			self.g = load_instance(filepath)
		else:

			self.g = g
			self.running_cost = np.inner(g.vp.cost.get_array(), 
				g.vp.is_upgraded.get_array())

		assert hasattr(self.g.vp, 'is_upgraded')
		assert hasattr(self.g.vp, 'cost')
		assert hasattr(self.g.ep, 'weight')
		assert hasattr(self.g.ep, 'all_weights')

		self.N = self.g.num_vertices()

		self.ewa = self.edge_weight_array()

		self._update_mst()


	def __str__(self):
		arr = self.g.vp.is_upgraded.get_array()
		return '{}, with obj_value {}'.format(arr, self.obj_value())


	def copy(self):
		return Solution(self.budget, g=self.g.copy())


	"""objective function value: weight of MST"""
	def obj_value(self):
		return self._obj_value


	def upgrade_vertex(self, v):

		g = self.g

		v = g.vertex(v)

		v_cost = g.vp.cost[v]

		if not g.vp.is_upgraded[v] and \
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

			self._update_mst()

			return True
		
		return False


	def downgrade_vertex(self, v):

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

			self._update_mst()

			return True	

		return False


	def _update_mst(self):

		g = self.g

		self.mst = min_spanning_tree(g, g.ep.weight)

		self._obj_value = sum(
			map(lambda e: g.ep.weight[e]*self.mst[e], g.edges())
		)


	# Gen array of edge weights with dimension (e,3).
	def edge_weight_array(self):
		ewa = np.zeros((self.g.num_edges(), 3))
		edges = self.g.get_edges()
		i = 0
		for e in edges:
			edge = self.g.edge(e[0], e[1])
			ewa[i] = np.array(self.g.ep.all_weights[edge])
			i += 1
		return ewa


	# Compute ratio of cost per upgrade.1
	#  note:iterating with iterators are slow compared to array operations
	def vertex_impact_ratio_on_tree(self):
		on_mst = self.mst.a.astype(bool)
		edges = self.g.get_edges()[on_mst]
		weights = self.g.ep.weight.a[on_mst]
		upgraded = np.ones(self.g.num_vertices(), dtype=bool) #self.g.vp.is_upgraded.a[edges[:,:2]]
		
		upgrade_level = np.sum(upgraded[edges[:,:2]], axis=1)
		delta = weights - self.ewa[on_mst, upgrade_level]

		# graph undirected -> edges needs to be accounted for both vertices
		delta = binned_statistic(
			np.concatenate((edges[:,0], edges[:,1])),
			np.concatenate((delta,delta)),
			statistic=np.sum,
			bins=np.arange(self.g.num_vertices() + 1))[0]

		return delta / self.g.vp.cost.a





class _NeighbourhoodIterator:


	def __init__(self, s, neigh, i=-1):
		
		self.s = s
		self.neigh = neigh
		self.i = i

		self.count = 0


	def __next__(self):

		if self.i < self.s.N:

			if self.i >= 0:
				self.s.downgrade_vertex(self.i)

			self.i += 1

			while self.i < self.s.N:

				if self.s.upgrade_vertex(self.i):
					self.neigh._mem[self.count] = self.i
					self.count += 1
					return self.s

				self.i += 1

		raise StopIteration


class Neighbourhood:


	"""Very basic neighbourhood for validation purposes"""
	def __init__(self, s):
		self.s = s
		self._mem = [-1] * s.N


	def __iter__(self):
		return _NeighbourhoodIterator(self.s.copy(), self)


	def __getitem__(self, i):
		"""returns i-th neighbour
		if not already calculated, 
		then the iterator is called until neighbour is obtained
		"""
		if i < 0 or i >= self.s.N:
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