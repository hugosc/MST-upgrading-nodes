import sys

import graph_tool.all as gt

from solution import Solution, Neighbourhood

f_p = sys.argv[1]
# print(sol.vertex_impact_ratio_on_tree())

from grasp import grasp
opt = grasp(Solution, [400, f_p], Neighbourhood)
# print(sol.g.vp.is_upgraded.a)

# graph = utils.load_instance(f_p)
# print(graph.num_vertices())
# tree_map = graph.new_edge_property("bool")

# gt.min_spanning_tree(graph, weights=graph.ep.weight, tree_map=tree_map)

# print(tree_map)