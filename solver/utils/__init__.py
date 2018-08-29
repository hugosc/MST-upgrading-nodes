__all__ = ["load_instance"]

from graph_tool.all import Graph
import matplotlib.pyplot as plt
import numpy as np

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
	edge_weight_lv2 = g.new_edge_property("float")
	edge_weight_lv3 = g.new_edge_property("float")

	edge_upgradeable_weights = g.new_edge_property("vector<float>")

	graph_total_cost = g.new_graph_property("float")
	graph_total_cost[g] = 0

	with open(filepath, "r") as f:

		first_line = f.readline()

		tokens = first_line.split(" ")

		assert len(tokens) == 2

		n = int(tokens[0])
		m = int(tokens[1])

		g.add_vertex(n)

		#print (n, m)

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
			edge_weight_lv2[e] = w2
			edge_weight_lv3[e] = w3

		# identify how weights are for vertices
		line = f.readline()
		v_cost = [ float(x) for x in line.split(" ") if x != " "]

		if len(v_cost) > 1: # case where formatting is incorrect
			for i in range(n):
				v = g.vertex(i)
				node_cost[v] = v_cost[i]
				node_upgraded[i] = False

		else:

			v = g.vertex(0)
			node_cost[v] = float(v_cost[0])
			node_upgraded[v] = False

			vertices = g.vertices()
			vertices.next()
			for v in vertices:

				line = f.readline()

				c = float(line)

				node_cost[v] = c
				node_upgraded[v] = False

	g.vp.is_upgraded   = node_upgraded
	g.vp.cost          = node_cost
	g.ep.weight        = edge_weight
	g.ep.weight_2      = edge_weight_lv2
	g.ep.weight_3      = edge_weight_lv3

	for v in g.vertices():
		graph_total_cost[g] += node_cost[v]
	g.gp.total_cost = graph_total_cost

	return g



def paths_to_files(f_names, path):
	p = path
	if path[-1] != "/":
		p += "/"
	return [p + f_name.replace('\n', '') for f_name in f_names]


# Plot mutliple curves sharing the same axis with different colors. If a tuple
#  of curves is an element of color_axis, all the curves inside that tuple will
#  have the same color.
def plot_color_curves(x_axis, color_axis, labels=None, fig_name="figure.png",
                      plot_title=None, x_label=None, y_label=None):
    num_curves = len(color_axis)
    colors = ['C' + str(c) for c in range(num_curves)]
    plots = list()

    for c in range(num_curves):
        curves = color_axis[c]
        c_color = colors[c]
        c_label = labels[c]
        if isinstance(curves, tuple):
            for curve in curves[1:]:
                plot, = plt.plot(x_axis, curve, color=c_color)
                plots.append(plot)
            curves = curves[0]
        plot, = plt.plot(x_axis, curves, color=c_color, label=c_label)
        plots.append(plot)

    if plot_title is not None:
        plt.title(plot_title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.legend(handles=plots)
    plt.grid(True) # coller
    plt.savefig(fig_name)
    plt.clf()




def generic_curves(color_axis, labels=None, fig_name="figure.png",
                      plot_title=None, x_label=None, y_label=None,
                      plot_function=plt.plot):
    fig = plt.figure()
    num_curves = len(color_axis)
    colors = ['C' + str(c) for c in range(num_curves)]
    plots = list()

    for c in range(num_curves):
        curves = color_axis[c]
        c_color = colors[c]
        c_label = labels[c]
        plots.append(plot_function(
                     curves[0], curves[1], color=c_color, label=c_label))

    if plot_title != None:
        plt.title(plot_title, y=1.16)
    if x_label != None:
        plt.xlabel(x_label)
    if y_label != None:
        plt.ylabel(y_label)
    plt.title(plot_title)#, y=1.16)
    plt.legend()
    plt.grid(True) # coller
    plt.savefig(fig_name)
