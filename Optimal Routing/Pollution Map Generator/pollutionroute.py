# TEST SCRIPT, DO NOT TRY TO USE LOL

import osmnx as ox
import networkx as nx
import numpy as np
import math


class Point:

    def __init__(self, y, x, value):
        self.y = y
        self.x = x
        self.value = value

    def __repr__(self):
        return f"[{self.y}, {self.x}, {self.value}]"

    def getY(self):
        return self.y

    def getX(self):
        return self.x

    def getValue(self):
        return self.value


city = "Terrassa"
G = ox.graph_from_place(city, network_type='drive')
Gnx = nx.relabel.convert_node_labels_to_integers(G)
nodes, edges = ox.graph_to_gdfs(Gnx, nodes=True, edges=True)

nodes['Pollution'] = float(0)

pollutionMatrix = np.loadtxt(open("map.csv", "rb"), delimiter=",", skiprows=1)

print(nodes)

print(max(nodes['y']))
print(min(nodes['y']))
print(max(nodes['x']))
print(min(nodes['x']))

rows = float(round(max(nodes['y']) - min(nodes['y']), 6))
cols = float(round(max(nodes['x']) - min(nodes['x']), 6))

reso = 100
incrow = round(rows / reso, 6)
incol = round(cols / reso, 6)

print(float(incrow), float(incol))
matrix = np.zeros((int(rows), int(cols)))  # np.zeros(int(rows),int(cols))

# Having troubles running through the matrix
points = []

y = min(nodes['y'])
x = max(nodes['x'])

# Completely wrong lmao
for row in range(len(pollutionMatrix)):
    for col in range(len(pollutionMatrix[row])):
        points.append(
            Point(round(y, 6), round(x, 6), pollutionMatrix[row][col]))
        if y >= max(nodes['y']):
            y = round(min(nodes['y']), 6)
        y = y + incol
        y = round(y, 6)
    if x <= min(nodes['x']):
        x = round(max(nodes['x']), 6)
    x = x - incrow
    x = round(x, 6)
    if (y >= max(nodes['y']) and x <= min(nodes['x'])):
        break
value = 0
selnode = 0
print(points)
pdist = 1000
first = True
for p in range(len(points)):
    x = points[p].getX()
    y = points[p].getY()
    point = tuple((float(y), float(x)))
    print(point)
    selectedNode, dist = ox.get_nearest_node(Gnx, point, return_dist=True)
    print(dist)
    print(points[p].getValue())
    if dist < 2000:
        nodes['Pollution'][int(selectedNode)] = float(points[p].getValue())
    if (float(dist) < float(pdist)) or first:
        pdist = dist
        selNode = selectedNode
        value = points[p].getValue()
    first = False
print(selNode)
nodes['Pollution'][int(selNode)] = float(value)

for node in range(len(nodes)):
    if nodes['Pollution'][node] > 0:
        print(nodes['Pollution'][node])

G2 = ox.graph_from_gdfs(nodes, edges)
origin_node = ox.get_nearest_node(G2, (41.56538214481594, 2.010169694600521))
destination_node = ox.get_nearest_node(G2,
                                       (41.56312304751537, 2.0302406158505706))
route = nx.shortest_path(G=G2,
                         source=origin_node,
                         target=destination_node,
                         weight='Pollution')
fig, ax = ox.plot_graph_route(
    G2,
    route,
    route_color="r",
    orig_dest_size=100,
    ax=None,
)
