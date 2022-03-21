import osmnx as ox
import networkx as nx
import taxicab as tc

G = ox.graph_from_place("Barcelona, Spain")
#projG = ox.project_graph(G)
#ox.plot_graph(projG)

origin_xy = (41.40025886794589, 2.1687767419461315)
destination_xy = (41.40378071100295, 2.1667991266734155)
origin_node = ox.get_nearest_node(G, origin_xy)
destination_node = ox.get_nearest_node(G, destination_xy)
route = nx.shortest_path(G=G,
                         source=origin_node,
                         target=destination_node,
                         weight='length')
routeTC = tc.distance.shortest_path(G, origin_xy, destination_xy)
fig, ax = tc.plot.plot_graph_route(G, routeTC)
