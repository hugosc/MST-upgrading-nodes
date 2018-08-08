from graph_tool.all import *
from graph_tool.topology import min_spanning_tree
from numpy import inner

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
			self.running_cost = inner(g.vp.cost.get_array(), 
				g.vp.is_upgraded.get_array())

		assert hasattr(self.g.vp, 'is_upgraded')
		assert hasattr(self.g.vp, 'cost')
		assert hasattr(self.g.ep, 'weight')
		assert hasattr(self.g.ep, 'all_weights')

		self.N = self.g.num_vertices()

		self._update_mst()


	def __str__(self):
		arr = self.g.vp.is_upgraded.get_array()
		return '{}, with obj_value {}'.format(arr, self.obj_value())


	def copy(self):

		return Solution(self.budget, g=self.g.copy())

	def obj_value(self):
		"""objective function value: weight of MST

		"""
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