import osmnx as ox
import networkx as nx
import numpy as np

city = "Terrassa"
G = ox.graph_from_place(city, network_type='drive')
Gnx = nx.relabel.convert_node_labels_to_integers(G)
nodes, edges = ox.graph_to_gdfs(Gnx, nodes=True, edges=True)

nodes['Pollution'] = 0

pollutionmat = np.loadtxt(open("map.csv", "rb"), delimiter=",", skiprows=1)

print(nodes)

# We need to sort the nodes by location. Once sorted, we can add the pollution values of the map to each node. Then remove the nodes that are above our pollution threshold.
# If removing nodes causes some kind of trouble, we should consider modifying networkx weighted shortestpath function in order to stablish a new rule.
for node in Gnx.nodes:
    for rows in pollutionmat[int(nodes)]:
        nodes['Pollution'][node] = pollutionmat[nodes][rows]