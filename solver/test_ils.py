from collections import deque
from numpy import inner

def gen_viables(budget, costs):

	n = len(costs)
	#queue of unviable solutions
	q = deque([[1]*n])

	forbidden = [False] * n

	while len(q):

		print('Forbidden %s and %s' % (forbidden, q))
		u = q.pop()

		for i in range(n):

			if (not forbidden[i]) and u[i]:
				v = u[:]
				v[i] = 0

				if inner(v, costs) <= budget:
					forbidden[i] = True
					yield v
				else:
					q.appendleft(v)


def gen_viables_(budget, costs):

	n = len(costs)

	solutions = set([tuple([0]* n)])
	maximals = set()

	while len(solutions):

		v = solutions.pop()

		is_maximal = True

		for i in range(n):

			if not v[i]:

				u = list(v)
				u[i] = 1

				if inner(u, costs) <= budget:

					is_maximal = False

					solutions.add(tuple(u))

		if is_maximal:

			t = tuple(v)
			if t not in maximals:
				yield v
				maximals.add(t)