import sys

# import graph_tool.all as gt

from solution import Solution, Neighbourhood

import matplotlib.pyplot as plt

import seaborn as sbn
import numpy as np
from grasp import ils
import time

f_p = sys.argv[1]

from grasp import Grasp
percentages = [0.1, 0.2, 0.3]
alphas = [0.2, 0.4, 0.6, 0.8]
p = 0.3

max_it = 100

obj_val = np.zeros((max_it, 4))

T = time.time()
grasp = Grasp(
	Solution, [lambda x: x * 0.3, f_p], Neighbourhood, alpha=0.4, max_it=100)

for i in range(max_it):
	j = 0
	for alpha in alphas:
		grasp.alpha = alpha
		obj_val[i][j] = grasp.build(grasp.solution).obj_value()
		j += 1



for i in range(4):
	fig,axis = plt.subplots()
	print(obj_val[:, i])
	sbn.distplot(obj_val[:, i], ax=axis, kde=True, hist=True)
	plt.title("Empirical distribution of solution values with noiseless a={}.".format(alphas[i]))
	plt.xlabel("Solution value.")
	plt.ylabel("Relative Frequency")
	plt.grid(True) # coller
	plt.savefig("alphakdeless{}.png".format(alphas[i]))
	plt.clf()