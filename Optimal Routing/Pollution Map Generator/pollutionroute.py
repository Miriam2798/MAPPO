# TEST SCRIPT, DO NOT TRY TO USE LOL

import osmnx as ox
import networkx as nx
import numpy as np
import math
import fastestpath as fp


#Defines a Point object with its own atributes Point(float y, float x, float PollutionValue)
class Point:

    def __init__(self, y, x, value):
        self.y = y
        self.x = x
        self.value = value

    #ToString function
    def __repr__(self):
        return f"[{self.y}, {self.x}, {self.value}]"

    #Getters
    def getY(self):
        return self.y

    def getX(self):
        return self.x

    def getValue(self):
        return self.value


city = input("Which city do you want to use? (Example: Barcelona) \n")
origin_yx = input("Origin point: \n")
destination_yx = input("Destination point: \n")
if (city == ""):
    city = "Terrassa"
if origin_yx == "" and destination_yx == "":
    origin_yx = (41.56538214481594, 2.010169694600521)
    destination_yx = (41.56312304751537, 2.0302406158505706)
G = fp.importFile(city)
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

#Iteration that gives geolocation values to every value in the pollution matrix. Objects Point(y,x,value) are stored into a points list
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

# Terrassa coordinates
#destination_node = ox.get_nearest_node(G2,
#origin_node = ox.get_nearest_node(G2, (41.56538214481594, 2.010169694600521))
#                                       (41.56312304751537, 2.0302406158505706))

# Barcelona Coordinates
origin_node = ox.get_nearest_node(G2, (origin_yx))
destination_node = ox.get_nearest_node(G2, (destination_yx))

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
