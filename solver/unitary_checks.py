import sys

import graph_tool.all as gt
import numpy as np

from solution import Solution, Neighbourhood

f_p = sys.argv[1]

from grasp import grasp
per = 0.5

sol = Solution(lambda x: per * x, f_p)
sol2 = sol.copy()
sol3 = sol.copy()

vertices = [0, 9, 4, 3]
sol.batch_vertex_upgrade(vertices)

for v in vertices:
    sol2.upgrade_vertex(v)

state = np.zeros(sol3.globals.N, dtype=bool)
state[vertices] = True
sol3.cleanse_to_state(state)

assert sol.edge_upgrade_level.__str__() == sol2.edge_upgrade_level.__str__(), \
        "both approaches should take you to the same solution"

assert sol.edge_upgrade_level.__str__() == sol3.edge_upgrade_level.__str__(), \
        "both approaches should take you to the same solution"
