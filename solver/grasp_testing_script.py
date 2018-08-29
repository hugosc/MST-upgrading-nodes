import sys

# import graph_tool.all as gt

from solution import Solution, Neighbourhood
from grasp import ils
import pandas as pd
import time

f_p = sys.argv[1]

from grasp import Grasp
percentages = [0.1, 0.2, 0.3]
for p in percentages:

	df = pd.DataFrame(columns=['tempo', 'qualidade'])
	
	for _ in range(10):
	

		print("percentage of total capacitance:" , p)

		grasp = Grasp(
			Solution, [lambda x: x * p, f_p], Neighbourhood, alpha=0.4, max_it=10)

		opt, t = grasp.run(ils, 2, 2, 6, 100)
		t_ = time.time()
		ils(opt, 2, 3, 7, 200)
		df = df.append({'tempo':t + (t_-time.time()), 'qualidade':opt.obj_value()}, ignore_index=True)
		print(opt)

	print(df.describe())
	print(df)