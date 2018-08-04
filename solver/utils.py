from graph_tool.all import Graph

#generic case, for sharing with class
def load_instance_(filepath):

	with open(filepath, "r") as f:

		first_line = f.readline()

		tokens = first_line.split(" ")

		assert len(tokens) == 2

		n = int(tokens[0])
		m = int(tokens[1])

		print (n, m)

		for _ in range(m):

			line = f.readline()

			tokens = line.split(" ")

			assert len(tokens) == 5

			v1 = int(tokens[0])
			v2 = int(tokens[1])

			w1 = float(tokens[2])
			w2 = float(tokens[3])
			w3 = float(tokens[4])

			print (v1, v2, w1, w2, w3)

		for _ in range(n):

			line = f.readline()

			c = float(line)

			print(c)

#our own version that loads a graph-tool Graph
def load_instance(filepath):

	g = Graph(directed=False)

	node_upgraded = g.new_vertex_property("bool")
	node_cost = g.new_vertex_property("float")

	edge_weight = g.new_edge_property("float")
	edge_upgradeable_weights = g.new_edge_property("vector<float>")
	edge_upgrade_level = g.new_edge_property("int")

	with open(filepath, "r") as f:

		first_line = f.readline()

		tokens = first_line.split(" ")

		assert len(tokens) == 2

		n = int(tokens[0])
		m = int(tokens[1])

		g.add_vertex(n)

		print (n, m)

		for _ in range(m):

			line = f.readline()

			tokens = line.split(" ")

			assert len(tokens) == 5

			v1 = int(tokens[0])
			v2 = int(tokens[1])

			e = g.add_edge(v1, v2)

			w1 = float(tokens[2])
			w2 = float(tokens[3])
			w3 = float(tokens[4])

			edge_weight[e] = w1
			edge_upgradeable_weights[e] = [w1, w2, w3]
			edge_upgrade_level[e] = 0

			print (v1, v2, w1, w2, w3)

		for v in g.vertices():

			line = f.readline()

			c = float(line)

			node_cost[v] = c
			node_upgraded[v] = False

	g.vp.is_upgraded   = node_upgraded
	g.vp.cost          = node_cost
	g.ep.weight        = edge_weight
	g.ep.all_weights   = edge_upgradeable_weights
	g.ep.upgrade_level = edge_upgrade_level

	return g

