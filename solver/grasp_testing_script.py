import sys

import graph_tool.all as gt

from solution import Solution, Neighbourhood

f_p = sys.argv[1]

from grasp import grasp
percentages = [0.1, 0.3, 0.5]
for p in percentages:
	print("percetage of total capacitance:" , p)
	opt = grasp(Solution, [lambda x: x * p, f_p], Neighbourhood)
