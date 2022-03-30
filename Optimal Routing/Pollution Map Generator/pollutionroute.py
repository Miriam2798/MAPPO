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

cols = int(round(max(nodes['x']) - min(nodes['x']), 6) * 100)
rows = int(round(max(nodes['y']) - min(nodes['y']), 6) * 100)

print(int(rows), int(cols))
matrix = np.zeros((int(rows), int(cols)))  # np.zeros(int(rows),int(cols))

# Having troubles running through the matrix
points = []

y = min(nodes['y'])
x = max(nodes['x'])

# Completely wrong lmao
for row in range(len(pollutionMatrix)):
    for col in range(len(pollutionMatrix[row])):
        points.append(Point(y, x, pollutionMatrix[row][col]))
        if y >= max(nodes['y']):
            y = min(nodes['y'])
        y = y + 1e-3
    if x <= min(nodes['x']):
        x = max(nodes['x'])
    x = x - 1e-3
    if (y >= max(nodes['y']) and x <= min(nodes['x'])):
        break
value = 0
selnode = 0
print(points)
pdist = 10000000
for p in range(len(points)):
    x = points[p].getX()
    y = points[p].getY()
    point = tuple((float(y), float(x)))
    print(point)
    selectedNode, dist = ox.get_nearest_node(G, point, return_dist=True)
    #print(dist)
    if dist < pdist:
        pdist = dist
        selNode = int(selectedNode)
        value = points[p].getValue()
        print(value)
nodes[selnode]['Pollution'] = float(value)
print(nodes)