import numpy as np
from scipy.stats import binned_statistic

from graph_tool.topology import min_spanning_tree as gt_min_spanning_tree
from utils import load_instance

from itertools import product, combinations

_01 = lambda x: 0.1*x
_03 = lambda x: 0.3*x
_04 = lambda x: 0.4*x
_05 = lambda x: 0.5*x
_09 = lambda x: 0.9*x

class SolutionGlobals:


	# everything that must be done ONLY once
	def __init__(self, budget, filepath=None, g=None):

		self.running_cost = 0
		
		assert filepath is not None or g is not None

		if g is not None:
			self.g = g
		else:
			self.g = load_instance(filepath)

		assert hasattr(self.g.vp, 'is_upgraded')
		assert hasattr(self.g.vp, 'cost')
		assert hasattr(self.g.ep, 'weight')
		assert hasattr(self.g.ep, 'weight_2')
		assert hasattr(self.g.ep, 'weight_3')


		self.N = self.g.num_vertices()
		self.E = self.g.num_edges()
		self.ewa = self.edge_weight_array()
		self.v_cost = self.g.vp.cost.a
		self.budget = budget(np.sum(self.v_cost))

		# graph tool returns edges not sorted by index
		#  note: to see something really crazy, uncoment lines bellow
		#
		self.edges = self.g.get_edges()
		self.edges = self.edges[self.edges[:,2].argsort()]

		self.v_degree = self.g.get_out_degrees(self.g.get_vertices())


	# Gen array of edge weights with dimension (e,3).
	#  Note: duplicate information from edges to make it faster
	def edge_weight_array(self):
		all_weights = np.column_stack((self.g.ep.weight.a,
									   self.g.ep.weight_2.a,
									   self.g.ep.weight_3.a))
		return np.vstack((all_weights, all_weights))


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

		self.cleanse_to_state(self.upgraded)

		self.edge_filter = self.globals.g.new_edge_property("boolean")
		self.mst_update_func = {True: self._fast_update_mst_upgrade,
								False: self._fast_update_mst_upgrade}


	# Compute which vertices are still able to be upgraded.
	def atualize_allowed_upgrades(self):
		self.to_upg = np.logical_and(
			np.logical_not(self.upgraded),
			self.globals.v_cost <= (self.globals.budget - self.running_cost))


	# Mix local information of upgraded vertices with global weight weights to
	#  find current edge weight values.
	def get_edge_weights(self):
		edges = self.globals.edges
		return self.globals.ewa[np.arange(self.globals.E, dtype=int),
			self.upgraded[edges[:, 0]].astype(int) + self.upgraded[edges[:, 1]]]


	# Rewind the solution state to a given state. When state is None, all 
	#  vertices are set to downgrade and every solution state attribute is set 
	#  as in object creation. The safe parameter can be used to cleanse to a
	#  forbidden state.
	#
	def cleanse_to_state(self, state=None, safe=True, mst=None):
		upg = state
		if upg is None:
			upg = np.zeros(self.globals.N, dtype=bool)

		r_cost = np.inner(self.globals.v_cost, upg)
		if safe and r_cost > self.globals.budget:
			return False

		self.upgraded = upg
		self.running_cost = np.inner(self.globals.v_cost, self.upgraded)
		self.cur_edge_weight = self.get_edge_weights()
		self.edge_upgrade_level = self.compute_edge_upgrade_level()
		self._DIRTY = False
		self._update_mst(mst=mst)
		self.atualize_allowed_upgrades()

		return True


	# Find edge upgrade level for current solution state. The incremental
	#  operation performed by fast_weight_update is more performatic when it
	#  can be done.
	#
	def compute_edge_upgrade_level(self):
		return self.upgraded[self.globals.edges[:,0]].astype(int) \
				+ self.upgraded[self.globals.edges[:,1]]


	# Pretty print for solution state.
	def __str__(self):
		arr = self.upgraded.astype(int)
		return '{}, with obj_value {} and cost {}'.format(
			arr, self.obj_value(), self.running_cost)


	# Create a new solution passing a reference for 'globals' singleton and
	#  reproducing the solution state of given vertices.
	#
	def copy(self):
		return Solution(sol_globals=self.globals, 
			sol_state=np.copy(self.upgraded))


	"""objective function value: weight of MST"""
	def obj_value(self):
		if self._DIRTY:
			self._update_mst()
			self._DIRTY = False

		return self._obj_value


	# Assumes you know what you're doing. Performs both upgrade and downgrade
	#  using mode selection. Gives speed assuming that it wont be called over
	#  to downgrade a non upgraded node or upgrade an already upgraded one.
	#
	def fast_v_upgrade(self, v, mode=True, update_mst=False):

		inc_mult = (mode * 1) + ((not mode) * -1)

		self.upgraded[v] = mode
		self.fast_weight_update(v, inc_mult)
		self.running_cost += inc_mult * self.globals.v_cost[v]

		if update_mst:
			# self.mst_update_func[mode](v)
			self._fast_update_mst_upgrade(v)
		else:
			self._DIRTY = True

		self.atualize_allowed_upgrades()


	# Control to change solution state of edge weights when a vertex is 
	#  upgraded.
	#
	def fast_weight_update(self, v, inc=1):
		e_indexes = self.globals.edges[:, 2][np.logical_or(
			self.globals.edges[:,0] == v,
			self.globals.edges[:, 1] == v)]
		self.edge_upgrade_level[e_indexes] += inc
		self.cur_edge_weight[e_indexes] = self.globals.ewa[e_indexes, 
			self.edge_upgrade_level[e_indexes]]


	# Batch version of fast_weight_update. Accepts a list of vertices. Works for
	#  increment, decrement of a mix of both, since will compute the levels 
	#  based on 'self.upgraded'.
	#
	def batch_weight_update(self, vertices):
		e_indexes = np.array([], dtype=int)

		# TODO: edges might come more than once on this iteration. Since the 
		#  value is being set in a compliant way, this isn't a problem for 
		#  correctness.
		for v in vertices:
			e_indexes = np.concatenate((
				e_indexes, 
				(self.globals.edges[:, 2][np.logical_or(
					self.globals.edges[:,0] == v,
					self.globals.edges[:, 1] == v)]).astype(int)))


		self.edge_upgrade_level[e_indexes] = \
			self.upgraded[self.globals.edges[e_indexes, 0]].astype(int) +\
			self.upgraded[self.globals.edges[e_indexes, 1]]

		self.cur_edge_weight[e_indexes] = self.globals.ewa[e_indexes, 
			self.edge_upgrade_level[e_indexes]]


	def upgrade_vertex(self, v, update_mst=False):

		if not self.upgraded[v] and \
		self.running_cost + self.globals.v_cost[v] <= self.globals.budget:
			self.fast_v_upgrade(v, update_mst=update_mst)
			return True

		return False


	# Batch upgrade of vertices. If the update_mst flag is unset, MST update
	#  will not achieve the speed up of "_batch_mst_update". This function only
	#  works for explicitly upgrades, vertex downgrades must be done with
	#  'cleanse_state' since MST will have to be rerun for all edges.
	#
	def batch_vertex_upgrade(self, vertices, update_mst=False):
		if not len(vertices):
			return

		self.upgraded[vertices] = True
		self.batch_weight_update(vertices)
		self.running_cost += np.sum(self.globals.v_cost[vertices])

		if update_mst:
			self._batch_mst_update(vertices)
		else:
			self._DIRTY = True

		self.atualize_allowed_upgrades()



	def downgrade_vertex(self, v, update_mst=False):

		if self.upgraded[v]:
			self.fast_v_upgrade(v, mode=False, update_mst=update_mst)
			return True
		return False


	# Upgrade a vertex even if the solution moves to an infeasible state.
	#
	def upgrade_vertex_unsafe(self, v, update_mst=False):

		if not self.upgraded[v]:
			self.fast_v_upgrade(v, mode=True, update_mst=update_mst)
			return True
		return False


	def _update_mst(self, v=None, mst=None):

		if mst is not None:
			self.mst = mst
		else:
			edge_weight = self.globals.g.new_edge_property("double")
			edge_weight.a = self.cur_edge_weight

			self.mst = gt_min_spanning_tree(self.globals.g, edge_weight)

		self._obj_value = self.total_tree_delay()


	def _fast_update_mst_upgrade(self, v):
		self.globals.g.ep.weight.a = self.cur_edge_weight

		self.mst.a[self.globals.g.get_out_edges(v)[:, 2]] = True

		self.globals.g.set_edge_filter(self.mst)
		self.mst = gt_min_spanning_tree(self.globals.g, self.globals.g.ep.weight)
		self.globals.g.set_edge_filter(None)

		self._obj_value = self.total_tree_delay()


	def _batch_mst_update(self, vertices):
		self.globals.g.ep.weight.a = self.cur_edge_weight

		self.mst.a[np.concatenate(np.array(
			[self.globals.g.get_out_edges(v)[:, 2] for v in vertices]))] = True

		self.globals.g.set_edge_filter(self.mst)
		self.mst = gt_min_spanning_tree(self.globals.g, self.globals.g.ep.weight)
		self.globals.g.set_edge_filter(None)

		self._obj_value = self.total_tree_delay()



	def total_tree_delay(self):
		return np.sum(self.cur_edge_weight[self.mst.a.astype(bool)])


	def available_vertices_to_upgrade(self):
		return np.sum(self.to_upg)


	def is_saturated(self):
		return self.available_vertices_to_upgrade() == 0



	# Compute current min_spam_tree upgrade cost per weight varition per vertex
	#  for all vertices simultaneously. Uses arrays as a promise of better
	#  performance from graph_tool documentation.
	#
	#  Note: iterating with iterators are slow compared to array operations.
	#
	def vertex_impact_ratio_on_tree(self):


		on_mst = self.mst.a.astype(bool)
		edges = self.globals.edges[on_mst]

		weights = self.cur_edge_weight[on_mst]

		e_to_upg_1 = self.to_upg[edges[:,0]]
		e_to_upg_2 = self.to_upg[edges[:,1]]

		e_indexes = np.concatenate((edges[e_to_upg_1, 2], edges[e_to_upg_2, 2]))


		# uses undirectedness of edges to represent different upgraded vertices
		edges = np.vstack((edges[:,[0,1]][e_to_upg_1], 
					edges[:,[1,0]][e_to_upg_2]))
		weights = np.concatenate((weights[e_to_upg_1], weights[e_to_upg_2]))
		delta = weights - self.globals.ewa[e_indexes,
		 self.to_upg[edges[:, 0]].astype(int) + self.upgraded[edges[:, 1]]]

		# compute delta for each edge and sort result by vertex label of first
		#  vertex on the edge(u in (u ,v))
		edge_impact = np.column_stack((edges, delta))
		edge_impact = edge_impact[edge_impact[:,0].argsort()]
		edge_impact[:, 2] = np.cumsum(edge_impact[:,2])

		selector = np.cumsum(
			np.unique(edge_impact[:, 0].astype(int), return_counts=True)[1]) - 1

		impact = edge_impact[selector, 2] - np.concatenate(
			([0], edge_impact[selector[:-1], 2]))

		return np.column_stack((impact / self.globals.v_cost[self.to_upg],
			   np.arange(self.globals.N)[self.to_upg]))



	# Generate a list of candidate vertices to act as perturbation points of the
	#  solution. Chooses 'n_total' vertices randomly, altough guarantees
	#  that 'n_actives' to be upgraded and 'n_incatives' to not.
	#
	def random_perturbation_candidates(self, n_actives, n_inactives, n_total):
		v = np.arange(self.globals.N)
		n_of_acs = n_actives + np.random.random_integers(
			n_total - n_actives - n_inactives)
		n_of_inacs = n_total - n_of_acs

		acs = np.random.choice(
			v[self.upgraded.astype(bool)], 
			size=n_of_acs, replace=False)

		inacs = np.random.choice(
			v[np.logical_not(self.upgraded.astype(bool))], 
			size=n_of_inacs, replace=False)

		return np.concatenate((acs, inacs))


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
		N = s.globals.N
		for comb in combinations(range(N), k):

			delta = 0
			exists_false = False

			for v in comb:
				if s.upgraded[v]:
					delta -= s.globals.v_cost[v]

				else:
					exists_false = True # TODO: never used variable
					delta += s.globals.v_cost[v]

			if s.running_cost + delta <= s.globals.budget and exists_false:
				yield comb


	def try_update(self, update_code):

		s = self.s

		for v in update_code:

			if not s.upgraded[v]:
				s.upgrade_vertex_unsafe(v)
			else:
				s.downgrade_vertex(v)

		obj_value = s.obj_value()

		for v in update_code:

			if not s.upgraded[v]:
				s.upgrade_vertex_unsafe(v)
			else:
				s.downgrade_vertex(v)

		return obj_value


	def update(self, update_code):

		s = self.s
 
		for v in update_code:
			if not s.upgraded[v]:
				s.upgrade_vertex_unsafe(v)
			else:
				s.downgrade_vertex(v)

		return s.obj_value()



def first_improvement(N, k):

	current_v = N.s.obj_value()
	for c in N.neighbour_codes(k):

		v = N.try_update(c)
		if v < current_v:

			N.update(c)
			return True

	return False

def grasp_vnd(Max_iteracoes,k,N,solucao_inicial):
    
      #k = 4 #máximo de elementos da solução que podem ser modificados
    
      #N = 1 # primeira estrutura de vizinhança
    
      vizinhanca = Neighbourhood(solucao_inicial)
    
   
      for i in range(Max_iteracoes): #enquanto critério de parada não for  satisfeito  
       
             while (N<=k): #permitido a troca de 4 estruturas de vizinhança
           
                 if  first_improvement(vizinhanca,N): # melhoria trocando os vértices
               
                     print (solucao_inicial)
                
                     N = 1 
            
                 else:    
                
                     N = N + 1
  
