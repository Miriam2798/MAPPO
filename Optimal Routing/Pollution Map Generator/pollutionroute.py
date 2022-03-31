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

print(int(max(nodes['y']) * 1000000))
print(min(nodes['y']))
print(max(nodes['x']) * 10000000)
print(min(nodes['x']))

rows = max(nodes['x']) * 100000 - min(nodes['x']) * 100000
cols = max(nodes['y']) * 10000 - min(nodes['y']) * 10000
print(int(rows), int(cols))
matrix = np.zeros((int(rows), int(cols)))  # np.zeros(int(rows),int(cols))

# Number of rows and cols from coordinates is too high. Currently not been able to map the coordinates with the values

for i in range(int(rows)):
    for j in range(int(cols)):
        matrix[i][j] = pollutionmat[i][j]

print(matrix)

# We need to sort the nodes by location. Once sorted, we can add the pollution values of the map to each node. Then remove the nodes that are above our pollution threshold.
# If removing nodes causes some kind of trouble, we should consider modifying networkx weighted shortestpath function in order to stablish a new rule.