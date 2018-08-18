from itertools import product, combinations
import numpy as np

class Neighbourhood_:

	def __init__(self, s):
		self.s = s

	def non_sub_bits(bits):

		for s in product([0, 1], repeat=len(bits)):

			if any(map(lambda b: b[0] and (not b[1]), zip(s, bits))):
				yield s


	def neighbour_codes(self, k):

		s = self.s

		for comb in combinations(range(s.N), k):

			g = s.g

			v_costs = np.array([g.vp.cost[i] for i in comb])

			upgraded = np.array([g.vp.is_upgraded[i] for i in comb])

			fixed_cost = s.running_cost - np.inner(v_costs, upgraded)

			for non_sub in Neighbourhood_.non_sub_bits(upgraded):
				
				if fixed_cost + np.inner(v_costs, non_sub) <= s.budget:
					yield comb, non_sub

	def try_update(self, update_code):

		s = self.s
 
		indices = update_code[0]
		flags   = update_code[1]

		tries = []

		for i, f in zip(indices, flags):

			if f:
				tries.append(s.upgrade_vertex(i))
			else:
				tries.append(s.downgrade_vertex(i))

		obj_value = s.obj_value()

		for i, f, t in zip(indices, flags, tries):

			if f and t:
				s.downgrade_vertex(i)
			elif t:
				s.upgrade_vertex(i)

		return obj_value

	def update(self, update_code):
		s = self.s
 
		indices = update_code[0]
		flags   = update_code[1]

		tries = []

		for i, f in zip(indices, flags):

			if f:
				tries.append(s.upgrade_vertex(i))
			else:
				tries.append(s.downgrade_vertex(i))

def local_improvement(N):
	best_v = N.s.obj_value()
	best_c = -1
		
	for c in N.neighbour_codes(2):
		v = N.try_update(c)
		if v < best_v:
			print(v, c)
			best_v = v
			best_c = c

	if best_c != -1:
		N.update(best_c)
