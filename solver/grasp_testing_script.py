import sys

import graph_tool.all as gt

from solution import Solution, Neighbourhood

f_p = sys.argv[1]

from grasp import Grasp
percentages = [0.1, 0.3, 0.5]
for p in percentages:
	print("percetage of total capacitance:" , p)
	grasp = Grasp(Solution, [lambda x: x * p, f_p], Neighbourhood, max_it=2)
	opt = grasp.run()
