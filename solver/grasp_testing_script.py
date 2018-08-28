import sys

# import graph_tool.all as gt

from solution import Solution, Neighbourhood
from grasp import ils

f_p = sys.argv[1]

from grasp import Grasp
percentages = [0.1, 0.2, 0.3]
for p in percentages:
	print("percetage of total capacitance:" , p)
	grasp = Grasp(
		Solution, [lambda x: x * p, f_p], Neighbourhood, alpha=0.4, max_it=1)
	opt = grasp.run(ils, 2, 2, 6, 100)

