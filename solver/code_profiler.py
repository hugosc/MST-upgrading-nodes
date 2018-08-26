import sys
import cProfile
import time

import graph_tool.all as gt

from solution import Solution, Neighbourhood
from utils import load_instance
from grasp import grasp

times = list()

f_p = sys.argv[1]
prof_f = sys.argv[2]
budgets = [0.1, 0.3, 0.5]
t = time.time()
g = load_instance(f_p)
times.append(time.time() - t)

def solve_for_multiple():
	percentages = [0.1, 0.3, 0.5]
	for p in percentages:
		# print("percetage of total capacitance:" , p)
		t = time.time()
		print(grasp(Solution, [lambda x: x * p, f_p, g], Neighbourhood))
		times.append(time.time() - t)

cProfile.run('solve_for_multiple()', prof_f)
print(times)